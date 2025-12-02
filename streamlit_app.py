import os
import json
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import requests
import ssl

# ----------------- CONFIG FROM ENV VARIABLES ----------------- #

DRIVE_UNLABELED_ID = os.environ["DRIVE_UNLABELED_ID"]
DRIVE_LABELED_ID   = os.environ["DRIVE_LABELED_ID"]
DRIVE_META_ID      = os.environ["DRIVE_META_ID"]
LABELS_FILENAME    = os.environ.get("LABELS_FILENAME", "labels.csv")

SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]

DRIVE_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

# ----------------- AUTH & LOW-LEVEL DRIVE HELPERS ----------------- #

@st.cache_resource
def get_drive_session():
    """Return an AuthorizedSession that can talk to Google Drive."""
    info = json.loads(SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return AuthorizedSession(creds)

def _drive_guard(fn, what: str):
    try:
        return fn()
    except (requests.RequestException, ssl.SSLError) as e:
        st.error(
            f"Error talking to Google Drive while **{what}**.\n\n"
            "This is a network/SSL issue between the host and Google. "
            "Please refresh the page or try again later.\n\n"
            f"Details: {e}"
        )
        st.stop()

def drive_list_files(session, q, fields="files(id,name)", page_size=1000):
    def _call():
        params = {
            "q": q,
            "fields": fields,
            "pageSize": page_size,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        r = session.get(f"{DRIVE_BASE}/files", params=params)
        r.raise_for_status()
        return r.json()
    return _drive_guard(_call, "listing files")

def drive_get_media(session, file_id):
    def _call():
        r = session.get(
            f"{DRIVE_BASE}/files/{file_id}",
            params={"alt": "media"},
            stream=True,
        )
        r.raise_for_status()
        return BytesIO(r.content)
    return _drive_guard(_call, f"downloading file {file_id}")

def drive_get_metadata(session, file_id, fields="id,parents"):
    def _call():
        r = session.get(
            f"{DRIVE_BASE}/files/{file_id}",
            params={"fields": fields},
        )
        r.raise_for_status()
        return r.json()
    return _drive_guard(_call, f"getting metadata for {file_id}")

def drive_update_media(session, file_id, data_bytes, mime_type="text/csv"):
    def _call():
        headers = {"Content-Type": mime_type}
        r = session.patch(
            f"{DRIVE_UPLOAD_BASE}/files/{file_id}",
            params={"uploadType": "media"},
            data=data_bytes,
            headers=headers,
        )
        r.raise_for_status()
        return r.json()
    return _drive_guard(_call, f"updating file {file_id}")

def drive_update_parents(session, file_id, add_parents, remove_parents):
    def _call():
        params = {
            "addParents": add_parents,
            "removeParents": remove_parents,
            "fields": "id,parents",
        }
        r = session.patch(f"{DRIVE_BASE}/files/{file_id}", params=params)
        r.raise_for_status()
        return r.json()
    return _drive_guard(_call, f"updating parents for {file_id}")

# ----------------- HIGH-LEVEL DRIVE OPERATIONS ----------------- #

def list_unlabeled_images(session):
    """Return all image files (id, name) in the unlabeled folder."""
    q = (
        f"'{DRIVE_UNLABELED_ID}' in parents "
        f"and mimeType contains 'image/' and trashed = false"
    )
    resp = drive_list_files(session, q, fields="files(id,name)", page_size=1000)
    return resp.get("files", [])

def get_next_unlabeled_image(session, labels_df):
    """
    Pick the first image in the unlabeled folder whose drive_file_id
    is NOT already in labels_df (i.e., not already labeled).
    """
    files = list_unlabeled_images(session)
    labeled_ids = set(labels_df["drive_file_id"].astype(str)) if not labels_df.empty else set()
    for f in files:
        if f["id"] not in labeled_ids:
            return f
    return None

def download_image_as_pil(session, file_id):
    fh = drive_get_media(session, file_id)
    img = Image.open(fh).convert("RGB")
    return img

def move_file_to_labeled(session, file_id):
    """
    Try to move file from unlabeled folder to labeled folder in Drive.
    If it fails, log a warning but DO NOT stop the app; the labeling
    queue is driven by labels.csv, not by folder membership.
    """
    try:
        meta = drive_get_metadata(session, file_id, fields="id,parents")
        prev_parents = ",".join(meta.get("parents", []))
        drive_update_parents(
            session,
            file_id,
            add_parents=DRIVE_LABELED_ID,
            remove_parents=prev_parents,
        )
    except Exception as e:
        st.warning(f"Could not move file {file_id} to labeled folder: {e}")

def get_labels_df(session):
    """
    Load labels.csv from Drive meta folder.
    Assumes labels.csv exists with the full schema.
    """
    q = f"'{DRIVE_META_ID}' in parents and trashed = false"
    resp = drive_list_files(session, q, fields="files(id,name)")
    files = resp.get("files", [])
    labels_files = [f for f in files if f["name"] == LABELS_FILENAME]

    if not labels_files:
        st.error(
            f"Could not find {LABELS_FILENAME} in the meta folder.\n\n"
            f"Please create it with the full header row for all structures."
        )
        st.stop()

    file_id = labels_files[0]["id"]
    fh = drive_get_media(session, file_id)

    cols = [
        "image_name","drive_file_id","annotator",
        "left_clip_bbox_x1","left_clip_bbox_y1","left_clip_bbox_x2","left_clip_bbox_y2",
        "left_clip_axis_x1","left_clip_axis_y1","left_clip_axis_x2","left_clip_axis_y2",
        "right_clip_bbox_x1","right_clip_bbox_y1","right_clip_bbox_x2","right_clip_bbox_y2",
        "right_clip_axis_x1","right_clip_axis_y1","right_clip_axis_x2","right_clip_axis_y2",
        "ant_leaflet_bbox_x1","ant_leaflet_bbox_y1","ant_leaflet_bbox_x2","ant_leaflet_bbox_y2",
        "post_leaflet_bbox_x1","post_leaflet_bbox_y1","post_leaflet_bbox_x2","post_leaflet_bbox_y2",
        "left_stem_bbox_x1","left_stem_bbox_y1","left_stem_bbox_x2","left_stem_bbox_y2",
        "left_stem_axis_x1","left_stem_axis_y1","left_stem_axis_x2","left_stem_axis_y2",
        "right_stem_bbox_x1","right_stem_bbox_y1","right_stem_bbox_x2","right_stem_bbox_y2",
        "right_stem_axis_x1","right_stem_axis_y1","right_stem_axis_x2","right_stem_axis_y2",
        "created_at",
    ]

    if fh.getbuffer().nbytes > 0:
        df = pd.read_csv(fh)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)

    return file_id, df

def save_labels_df(session, file_id, df):
    data = df.to_csv(index=False).encode("utf-8")
    drive_update_media(session, file_id, data, mime_type="text/csv")

# ----------------- STREAMLIT APP LAYOUT ----------------- #

st.set_page_config(page_title="TEE Clip Labeling", layout="wide")

session = get_drive_session()

# Cache labels in session_state (Option 1)
if "labels_df" not in st.session_state:
    labels_file_id, labels_df = get_labels_df(session)
    st.session_state["labels_df"] = labels_df
    st.session_state["labels_file_id"] = labels_file_id
else:
    labels_df = st.session_state["labels_df"]
    labels_file_id = st.session_state["labels_file_id"]

st.sidebar.title("Annotator")
annotator = st.sidebar.text_input("Enter your ID/name", value="", max_chars=50)
if not annotator:
    st.sidebar.warning("Please enter your annotator ID to start labeling.")

page = st.sidebar.radio("Page", ["Dashboard", "Labeling"])

# ----------------- DASHBOARD PAGE ----------------- #

if page == "Dashboard":
    st.title("Labeling Dashboard")

    all_unlabeled_files = list_unlabeled_images(session)
    labeled_ids = set(labels_df["drive_file_id"].astype(str)) if not labels_df.empty else set()
    total_in_folder = len(all_unlabeled_files)
    total_unlabeled = sum(1 for f in all_unlabeled_files if f["id"] not in labeled_ids)
    total_labeled = len(labels_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Images in unlabeled folder", total_in_folder)
    with c2:
        st.metric("Unlabeled images (not in CSV)", total_unlabeled)
    with c3:
        st.metric("Labeled images (rows in CSV)", total_labeled)

    if total_labeled > 0:
        st.subheader("Labels per annotator")
        annot_counts = labels_df["annotator"].value_counts().reset_index()
        annot_counts.columns = ["annotator", "count"]
        st.table(annot_counts)

        st.subheader("Recent labels")
        st.dataframe(labels_df.sort_values("created_at", ascending=False).head(20))

# ----------------- LABELING PAGE (single canvas, multi-feature) ----------------- #

if page == "Labeling":
    st.title("TEE Clip & Leaflet Labeling")

    if not annotator:
        st.warning("Please enter your annotator ID in the sidebar.")
        st.stop()

    # Pick or remember current image
    if "current_file_id" not in st.session_state:
        current_file = get_next_unlabeled_image(session, labels_df)
        if current_file is None:
            st.success("No more unlabeled images left to label in this folder.")
            st.stop()
        st.session_state["current_file_id"] = current_file["id"]
        st.session_state["current_image_name"] = current_file["name"]
        st.session_state["canvas_json"] = None  # no existing drawings yet

    file_id = st.session_state["current_file_id"]
    image_name = st.session_state["current_image_name"]

    # Load and resize image
    img = download_image_as_pil(session, file_id)
    orig_width, orig_height = img.size

    max_dim = 900
    scale = 1.0
    if max(orig_width, orig_height) > max_dim:
        scale = max_dim / max(orig_width, orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img = img.resize((new_width, new_height))
        width, height = new_width, new_height
    else:
        width, height = orig_width, orig_height

    st.write(f"Displayed image size: {width} x {height}  (scale factor {scale:.3f})")

    # Store scale so we can unscale later
    st.session_state["current_scale"] = scale

    # ---- Feature configuration ---- #

    FEATURES = {
        "left_clip": {
            "label": "Left clip",
            "color": "#FF0000",   # red
            "has_axis": True,
        },
        "right_clip": {
            "label": "Right clip",
            "color": "#0000FF",   # blue
            "has_axis": True,
        },
        "ant_leaflet": {
            "label": "Anterior leaflet",
            "color": "#00FF00",   # green
            "has_axis": False,
        },
        "post_leaflet": {
            "label": "Posterior leaflet",
            "color": "#FFA500",   # orange
            "has_axis": False,
        },
        "left_stem": {
            "label": "Left clip stem",
            "color": "#FF00FF",   # magenta
            "has_axis": True,
        },
        "right_stem": {
            "label": "Right clip stem",
            "color": "#00FFFF",   # cyan
            "has_axis": True,
        },
    }

    # Current active tool (feature + mode)
    if "current_feature" not in st.session_state:
        st.session_state["current_feature"] = "left_clip"
    if "current_mode" not in st.session_state:
        st.session_state["current_mode"] = "rect"  # "rect" or "line"

    def set_tool(feature_key, mode):
        st.session_state["current_feature"] = feature_key
        st.session_state["current_mode"] = mode

    def clear_feature(feature_key):
        """Remove all shapes for this feature's color from the stored canvas JSON."""
        color = FEATURES[feature_key]["color"]
        data = st.session_state.get("canvas_json")
        if not data or "objects" not in data:
            return
        new_objs = [obj for obj in data["objects"] if obj.get("stroke") != color]
        data["objects"] = new_objs
        st.session_state["canvas_json"] = data

    # ---- Layout: controls on the left, canvas on the right ---- #

    col_controls, col_canvas = st.columns([1, 3])

    with col_controls:
        st.subheader("Tools")

        for key, meta in FEATURES.items():
            st.markdown(f"**{meta['label']}**  \n"
                        f"<span style='color:{meta['color']}'>■</span> annotations",
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 1])

            # Rectangle button
            if c1.button("Rect", key=f"btn_rect_{key}"):
                set_tool(key, "rect")

            # Axis (line) button if this feature has an axis
            if meta["has_axis"]:
                if c2.button("Axis", key=f"btn_axis_{key}"):
                    set_tool(key, "line")
                clear_col = c3
            else:
                clear_col = c2

            # Clear button (removes shapes for this feature only)
            if clear_col.button("Clear", key=f"btn_clear_{key}"):
                clear_feature(key)
                st.experimental_rerun()

        st.markdown("---")
        if st.button("Submit labels", key="submit_labels"):
            canvas_json = st.session_state.get("canvas_json")
            if not canvas_json or "objects" not in canvas_json:
                st.error("No annotations found on the canvas.")
            else:
                # Extract per-feature rect & axis from all objects
                objects = canvas_json["objects"]
                feature_rect = {k: None for k in FEATURES.keys()}
                feature_axis = {k: None for k in FEATURES.keys()}

                for key, meta in FEATURES.items():
                    color = meta["color"]
                    for obj in objects:
                        if obj.get("stroke") != color:
                            continue
                        if obj["type"] == "rect":
                            left = obj["left"]; top = obj["top"]
                            w = obj["width"]; h = obj["height"]
                            feature_rect[key] = (left, top, left + w, top + h)
                        elif obj["type"] == "line":
                            feature_axis[key] = (
                                obj["x1"], obj["y1"], obj["x2"], obj["y2"]
                            )

                scale = st.session_state.get("current_scale", 1.0)

                def unscale_bbox(b):
                    if b is None:
                        return (None, None, None, None)
                    if scale != 1.0:
                        inv = 1.0 / scale
                        return (b[0]*inv, b[1]*inv, b[2]*inv, b[3]*inv)
                    else:
                        return b

                def unscale_line(l):
                    if l is None:
                        return (None, None, None, None)
                    if scale != 1.0:
                        inv = 1.0 / scale
                        return (l[0]*inv, l[1]*inv, l[2]*inv, l[3]*inv)
                    else:
                        return l

                # Unscale everything to original image coordinates
                lc_bbox = unscale_bbox(feature_rect["left_clip"])
                lc_axis = unscale_line(feature_axis["left_clip"])
                rc_bbox = unscale_bbox(feature_rect["right_clip"])
                rc_axis = unscale_line(feature_axis["right_clip"])
                ant_bbox = unscale_bbox(feature_rect["ant_leaflet"])
                post_bbox = unscale_bbox(feature_rect["post_leaflet"])
                ls_bbox = unscale_bbox(feature_rect["left_stem"])
                ls_axis = unscale_line(feature_axis["left_stem"])
                rs_bbox = unscale_bbox(feature_rect["right_stem"])
                rs_axis = unscale_line(feature_axis["right_stem"])

                new_row = {
                    "image_name": image_name,
                    "drive_file_id": file_id,
                    "annotator": annotator,
                    "left_clip_bbox_x1": lc_bbox[0],
                    "left_clip_bbox_y1": lc_bbox[1],
                    "left_clip_bbox_x2": lc_bbox[2],
                    "left_clip_bbox_y2": lc_bbox[3],
                    "left_clip_axis_x1": lc_axis[0],
                    "left_clip_axis_y1": lc_axis[1],
                    "left_clip_axis_x2": lc_axis[2],
                    "left_clip_axis_y2": lc_axis[3],
                    "right_clip_bbox_x1": rc_bbox[0],
                    "right_clip_bbox_y1": rc_bbox[1],
                    "right_clip_bbox_x2": rc_bbox[2],
                    "right_clip_bbox_y2": rc_bbox[3],
                    "right_clip_axis_x1": rc_axis[0],
                    "right_clip_axis_y1": rc_axis[1],
                    "right_clip_axis_x2": rc_axis[2],
                    "right_clip_axis_y2": rc_axis[3],
                    "ant_leaflet_bbox_x1": ant_bbox[0],
                    "ant_leaflet_bbox_y1": ant_bbox[1],
                    "ant_leaflet_bbox_x2": ant_bbox[2],
                    "ant_leaflet_bbox_y2": ant_bbox[3],
                    "post_leaflet_bbox_x1": post_bbox[0],
                    "post_leaflet_bbox_y1": post_bbox[1],
                    "post_leaflet_bbox_x2": post_bbox[2],
                    "post_leaflet_bbox_y2": post_bbox[3],
                    "left_stem_bbox_x1": ls_bbox[0],
                    "left_stem_bbox_y1": ls_bbox[1],
                    "left_stem_bbox_x2": ls_bbox[2],
                    "left_stem_bbox_y2": ls_bbox[3],
                    "left_stem_axis_x1": ls_axis[0],
                    "left_stem_axis_y1": ls_axis[1],
                    "left_stem_axis_x2": ls_axis[2],
                    "left_stem_axis_y2": ls_axis[3],
                    "right_stem_bbox_x1": rs_bbox[0],
                    "right_stem_bbox_y1": rs_bbox[1],
                    "right_stem_bbox_x2": rs_bbox[2],
                    "right_stem_bbox_y2": rs_bbox[3],
                    "right_stem_axis_x1": rs_axis[0],
                    "right_stem_axis_y1": rs_axis[1],
                    "right_stem_axis_x2": rs_axis[2],
                    "right_stem_axis_y2": rs_axis[3],
                    "created_at": datetime.utcnow().isoformat(),
                }

                labels_df = pd.concat(
                    [labels_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_labels_df(session, labels_file_id, labels_df)

                st.session_state["labels_df"] = labels_df
                st.session_state["labels_file_id"] = labels_file_id

                move_file_to_labeled(session, file_id)

                # Reset for next image
                st.session_state.pop("current_file_id", None)
                st.session_state.pop("current_image_name", None)
                st.session_state.pop("canvas_json", None)

                st.success("Labels saved! Loading next image...")
                st.experimental_rerun()

    # ---- Canvas on the right ---- #

    with col_canvas:
        st.subheader(f"Image: {image_name}")

        # Current drawing mode & color
        active_feature = st.session_state["current_feature"]
        active_mode = st.session_state["current_mode"]
        active_color = FEATURES[active_feature]["color"]
        drawing_mode = "rect" if active_mode == "rect" else "line"

        st.caption(
            f"Active tool: {FEATURES[active_feature]['label']} – "
            f"{'Rectangle' if drawing_mode == 'rect' else 'Axis line'}"
        )

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color=active_color,
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key=f"canvas_main_{file_id}",
            initial_drawing=st.session_state.get("canvas_json"),
            display_toolbar=True,
        )

        # Persist the latest canvas JSON so we can submit / clear later
        if canvas_result.json_data is not None:
            st.session_state["canvas_json"] = canvas_result.json_data

