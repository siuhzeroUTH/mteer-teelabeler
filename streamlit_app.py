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

# ----------------- LABELING PAGE (6-step wizard, with persistent canvas + Clear) ----------------- #

if page == "Labeling":
    st.title("TEE Clip & Leaflet Labeling")

    if not annotator:
        st.warning("Please enter your annotator ID in the sidebar.")
        st.stop()

    # Wizard state
    if "label_step" not in st.session_state:
        st.session_state["label_step"] = 1

    # Pick or remember current image
    if "current_file_id" not in st.session_state:
        current_file = get_next_unlabeled_image(session, labels_df)
        if current_file is None:
            st.success("No more unlabeled images left to label in this folder.")
            st.stop()
        st.session_state["current_file_id"] = current_file["id"]
        st.session_state["current_image_name"] = current_file["name"]
        st.session_state["raw_shapes"] = {}  # will hold raw bboxes/axes for this image
        # also clear any old canvas data from previous image
        for k in list(st.session_state.keys()):
            if k.startswith("canvas_step"):
                del st.session_state[k]

    file_id = st.session_state["current_file_id"]
    image_name = st.session_state["current_image_name"]
    step = st.session_state["label_step"]
    raw_shapes = st.session_state["raw_shapes"]

    st.subheader(f"Image: {image_name}  |  Step {step} of 6")

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

    st.write(f"Displayed image size: {width} x {height} (scale factor {scale:.3f})")

    # helpers for scaling
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

    def parse_first_rect_and_line(json_data):
        bbox = None
        axis = None
        if json_data is not None:
            for obj in json_data.get("objects", []):
                if obj["type"] == "rect" and bbox is None:
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    bbox = (left, top, left + w, top + h)
                elif obj["type"] == "line" and axis is None:
                    axis = (obj["x1"], obj["y1"], obj["x2"], obj["y2"])
        return bbox, axis

    def parse_first_rect(json_data):
        bbox = None
        if json_data is not None:
            for obj in json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    bbox = (left, top, left + w, top + h)
                    break
        return bbox

    # small helper to get per-step canvas key
    def canvas_key(step_num):
        return f"canvas_step{step_num}_{file_id}"

    # ------- Step 1: Left clip (bbox + axis) ------- #
    if step == 1:
        st.markdown("### Step 1: Left clip body")

        tool = st.radio(
            "Choose drawing tool:",
            ["Rectangle (bbox)", "Axis (line)"],
            horizontal=True,
            key=f"tool_step1_{file_id}",
        )
        draw_mode = "rect" if tool.startswith("Rectangle") else "line"

        ckey = canvas_key(1)
        initial = st.session_state.get(ckey)

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#FF0000",  # red
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=draw_mode,
            key=f"canvas_step1_{file_id}",
            initial_drawing=initial,
            display_toolbar=True,
        )

        if canvas.json_data is not None:
            st.session_state[ckey] = canvas.json_data

        left_clip_bbox, left_clip_axis = parse_first_rect_and_line(
            st.session_state.get(ckey)
        )

        cols = st.columns(3)
        if cols[0].button("Clear drawing"):
            st.session_state[ckey] = None
            st.rerun()
        if cols[1].button("Next (right clip)"):
            if left_clip_bbox is None or left_clip_axis is None:
                st.error("Please draw BOTH a rectangle and a line for the left clip.")
            else:
                raw_shapes["left_clip_bbox"] = left_clip_bbox
                raw_shapes["left_clip_axis"] = left_clip_axis
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 2
                st.rerun()

    # ------- Step 2: Right clip (bbox + axis) ------- #
    elif step == 2:
        st.markdown("### Step 2: Right clip body")

        tool = st.radio(
            "Choose drawing tool:",
            ["Rectangle (bbox)", "Axis (line)"],
            horizontal=True,
            key=f"tool_step2_{file_id}",
        )
        draw_mode = "rect" if tool.startswith("Rectangle") else "line"

        ckey = canvas_key(2)
        initial = st.session_state.get(ckey)

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#0000FF",  # blue
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=draw_mode,
            key=f"canvas_step2_{file_id}",
            initial_drawing=initial,
            display_toolbar=True,
        )

        if canvas.json_data is not None:
            st.session_state[ckey] = canvas.json_data

        right_clip_bbox, right_clip_axis = parse_first_rect_and_line(
            st.session_state.get(ckey)
        )

        cols = st.columns(3)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 1
            st.rerun()
        if cols[1].button("Clear drawing"):
            st.session_state[ckey] = None
            st.rerun()
        if cols[2].button("Next (anterior leaflet)"):
            if right_clip_bbox is None or right_clip_axis is None:
                st.error("Please draw BOTH a rectangle and a line for the right clip.")
            else:
                raw_shapes["right_clip_bbox"] = right_clip_bbox
                raw_shapes["right_clip_axis"] = right_clip_axis
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 3
                st.rerun()

    # ------- Step 3: Anterior leaflet (bbox only) ------- #
    elif step == 3:
        st.markdown("### Step 3: Anterior leaflet")

        ckey = canvas_key(3)
        initial = st.session_state.get(ckey)

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#00FF00",  # green
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="rect",
            key=f"canvas_step3_{file_id}",
            initial_drawing=initial,
            display_toolbar=True,
        )

        if canvas.json_data is not None:
            st.session_state[ckey] = canvas.json_data

        ant_leaflet_bbox = parse_first_rect(st.session_state.get(ckey))

        cols = st.columns(3)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 2
            st.rerun()
        if cols[1].button("Clear drawing"):
            st.session_state[ckey] = None
            st.rerun()
        if cols[2].button("Next (posterior leaflet)"):
            if ant_leaflet_bbox is None:
                st.error("Please draw a rectangle for the anterior leaflet.")
            else:
                raw_shapes["ant_leaflet_bbox"] = ant_leaflet_bbox
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 4
                st.rerun()

    # ------- Step 4: Posterior leaflet (bbox only) ------- #
    elif step == 4:
        st.markdown("### Step 4: Posterior leaflet")

        ckey = canvas_key(4)
        initial = st.session_state.get(ckey)

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#FFA500",  # orange
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="rect",
            key=f"canvas_step4_{file_id}",
            initial_drawing=initial,
            display_toolbar=True,
        )

        if canvas.json_data is not None:
            st.session_state[ckey] = canvas.json_data

        post_leaflet_bbox = parse_first_rect(st.session_state.get(ckey))

        cols = st.columns(3)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 3
            st.rerun()
        if cols[1].button("Clear drawing"):
            st.session_state[ckey] = None
            st.rerun()
        if cols[2].button("Next (left clip stem)"):
            if post_leaflet_bbox is None:
                st.error("Please draw a rectangle for the posterior leaflet.")
            else:
                raw_shapes["post_leaflet_bbox"] = post_leaflet_bbox
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 5
                st.rerun()

    # ------- Step 5: Left clip stem (bbox + axis) ------- #
    elif step == 5:
        st.markdown("### Step 5: Left clip stem")

        tool = st.radio(
            "Choose drawing tool:",
            ["Rectangle (bbox)", "Axis (line)"],
            horizontal=True,
            key=f"tool_step5_{file_id}",
        )
        draw_mode = "rect" if tool.startswith("Rectangle") else "line"

        ckey = canvas_key(5)
        initial = st.session_state.get(ckey)

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#FF00FF",  # magenta
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=draw_mode,
            key=f"canvas_step5_{file_id}",
            initial_drawing=initial,
            display_toolbar=True,
        )

        if canvas.json_data is not None:
            st.session_state[ckey] = canvas.json_data

        left_stem_bbox, left_stem_axis = parse_first_rect_and_line(
            st.session_state.get(ckey)
        )

        cols = st.columns(3)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 4
            st.rerun()
        if cols[1].button("Clear drawing"):
            st.session_state[ckey] = None
            st.rerun()
        if cols[2].button("Next (right clip stem)"):
            if left_stem_bbox is None or left_stem_axis is None:
                st.error("Please draw BOTH a rectangle and a line for the left stem.")
            else:
                raw_shapes["left_stem_bbox"] = left_stem_bbox
                raw_shapes["left_stem_axis"] = left_stem_axis
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 6
                st.rerun()

    # ------- Step 6: Right clip stem (bbox + axis) + submit ------- #
    elif step == 6:
        st.markdown("### Step 6: Right clip stem")

        tool = st.radio(
            "Choose drawing tool:",
            ["Rectangle (bbox)", "Axis (line)"],
            horizontal=True,
            key=f"tool_step6_{file_id}",
        )
        draw_mode = "rect" if tool.startswith("Rectangle") else "line"

        ckey = canvas_key(6)
        initial = st.session_state.get(ckey)

        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#00FFFF",  # cyan
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=draw_mode,
            key=f"canvas_step6_{file_id}",
            initial_drawing=initial,
            display_toolbar=True,
        )

        if canvas.json_data is not None:
            st.session_state[ckey] = canvas.json_data

        right_stem_bbox, right_stem_axis = parse_first_rect_and_line(
            st.session_state.get(ckey)
        )

        cols = st.columns(3)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 5
            st.rerun()
        if cols[1].button("Clear drawing"):
            st.session_state[ckey] = None
            st.rerun()

        if cols[2].button("Submit all labels for this image"):
            if right_stem_bbox is None or right_stem_axis is None:
                st.error("Please draw BOTH a rectangle and a line for the right stem.")
            else:
                raw_shapes["right_stem_bbox"] = right_stem_bbox
                raw_shapes["right_stem_axis"] = right_stem_axis

                # extract and unscale all shapes
                left_clip_bbox   = raw_shapes.get("left_clip_bbox")
                left_clip_axis   = raw_shapes.get("left_clip_axis")
                right_clip_bbox  = raw_shapes.get("right_clip_bbox")
                right_clip_axis  = raw_shapes.get("right_clip_axis")
                ant_leaflet_bbox = raw_shapes.get("ant_leaflet_bbox")
                post_leaflet_bbox= raw_shapes.get("post_leaflet_bbox")
                left_stem_bbox   = raw_shapes.get("left_stem_bbox")
                left_stem_axis   = raw_shapes.get("left_stem_axis")
                right_stem_bbox  = raw_shapes.get("right_stem_bbox")
                right_stem_axis  = raw_shapes.get("right_stem_axis")

                lc_x1, lc_y1, lc_x2, lc_y2 = unscale_bbox(left_clip_bbox)
                lca_x1, lca_y1, lca_x2, lca_y2 = unscale_line(left_clip_axis)
                rc_x1, rc_y1, rc_x2, rc_y2 = unscale_bbox(right_clip_bbox)
                rca_x1, rca_y1, rca_x2, rca_y2 = unscale_line(right_clip_axis)
                ant_x1, ant_y1, ant_x2, ant_y2 = unscale_bbox(ant_leaflet_bbox)
                post_x1, post_y1, post_x2, post_y2 = unscale_bbox(post_leaflet_bbox)
                lst_x1, lst_y1, lst_x2, lst_y2 = unscale_bbox(left_stem_bbox)
                lsta_x1, lsta_y1, lsta_x2, lsta_y2 = unscale_line(left_stem_axis)
                rst_x1, rst_y1, rst_x2, rst_y2 = unscale_bbox(right_stem_bbox)
                rsta_x1, rsta_y1, rsta_x2, rsta_y2 = unscale_line(right_stem_axis)

                new_row = {
                    "image_name": image_name,
                    "drive_file_id": file_id,
                    "annotator": annotator,
                    "left_clip_bbox_x1": lc_x1,
                    "left_clip_bbox_y1": lc_y1,
                    "left_clip_bbox_x2": lc_x2,
                    "left_clip_bbox_y2": lc_y2,
                    "left_clip_axis_x1": lca_x1,
                    "left_clip_axis_y1": lca_y1,
                    "left_clip_axis_x2": lca_x2,
                    "left_clip_axis_y2": lca_y2,
                    "right_clip_bbox_x1": rc_x1,
                    "right_clip_bbox_y1": rc_y1,
                    "right_clip_bbox_x2": rc_x2,
                    "right_clip_bbox_y2": rc_y2,
                    "right_clip_axis_x1": rca_x1,
                    "right_clip_axis_y1": rca_y1,
                    "right_clip_axis_x2": rca_x2,
                    "right_clip_axis_y2": rca_y2,
                    "ant_leaflet_bbox_x1": ant_x1,
                    "ant_leaflet_bbox_y1": ant_y1,
                    "ant_leaflet_bbox_x2": ant_x2,
                    "ant_leaflet_bbox_y2": ant_y2,
                    "post_leaflet_bbox_x1": post_x1,
                    "post_leaflet_bbox_y1": post_y1,
                    "post_leaflet_bbox_x2": post_x2,
                    "post_leaflet_bbox_y2": post_y2,
                    "left_stem_bbox_x1": lst_x1,
                    "left_stem_bbox_y1": lst_y1,
                    "left_stem_bbox_x2": lst_x2,
                    "left_stem_bbox_y2": lst_y2,
                    "left_stem_axis_x1": lsta_x1,
                    "left_stem_axis_y1": lsta_y1,
                    "left_stem_axis_x2": lsta_x2,
                    "left_stem_axis_y2": lsta_y2,
                    "right_stem_bbox_x1": rst_x1,
                    "right_stem_bbox_y1": rst_y1,
                    "right_stem_bbox_x2": rst_x2,
                    "right_stem_bbox_y2": rst_y2,
                    "right_stem_axis_x1": rsta_x1,
                    "right_stem_axis_y1": rsta_y1,
                    "right_stem_axis_x2": rsta_x2,
                    "right_stem_axis_y2": rsta_y2,
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

                # reset wizard & canvas for next image
                st.session_state.pop("current_file_id", None)
                st.session_state.pop("current_image_name", None)
                st.session_state.pop("raw_shapes", None)
                for k in list(st.session_state.keys()):
                    if k.startswith("canvas_step"):
                        del st.session_state[k]
                st.session_state["label_step"] = 1

                st.success("Labels saved! Loading next image...")
                st.rerun()


