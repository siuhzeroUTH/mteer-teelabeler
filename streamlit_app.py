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
        "clip_bbox_x1","clip_bbox_y1","clip_bbox_x2","clip_bbox_y2",
        "axis_x1","axis_y1","axis_x2","axis_y2",
        "left_clip_bbox_x1","left_clip_bbox_y1","left_clip_bbox_x2","left_clip_bbox_y2",
        "right_clip_bbox_x1","right_clip_bbox_y1","right_clip_bbox_x2","right_clip_bbox_y2",
        "ant_leaflet_bbox_x1","ant_leaflet_bbox_y1","ant_leaflet_bbox_x2","ant_leaflet_bbox_y2",
        "post_leaflet_bbox_x1","post_leaflet_bbox_y1","post_leaflet_bbox_x2","post_leaflet_bbox_y2",
        "left_stem_bbox_x1","left_stem_bbox_y1","left_stem_bbox_x2","left_stem_bbox_y2",
        "right_stem_bbox_x1","right_stem_bbox_y1","right_stem_bbox_x2","right_stem_bbox_y2",
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

# ----------------- LABELING PAGE (wizard) ----------------- #

if page == "Labeling":
    st.title("TEE Clip & Leaflet Labeling")

    if not annotator:
        st.warning("Please enter your annotator ID in the sidebar.")
        st.stop()

    # Wizard state
    if "label_step" not in st.session_state:
        st.session_state["label_step"] = 1

    if "current_file_id" not in st.session_state:
        current_file = get_next_unlabeled_image(session, labels_df)
        if current_file is None:
            st.success("No more unlabeled images left to label in this folder.")
            st.stop()
        st.session_state["current_file_id"] = current_file["id"]
        st.session_state["current_image_name"] = current_file["name"]
        st.session_state["raw_shapes"] = {}  # will hold raw bboxes/axis in canvas coords

    file_id = st.session_state["current_file_id"]
    image_name = st.session_state["current_image_name"]
    step = st.session_state["label_step"]
    raw_shapes = st.session_state["raw_shapes"]

    st.subheader(f"Image: {image_name}  |  Step {step} of 7")
    st.write("DEBUG – current file id:", file_id)

    img = download_image_as_pil(session, file_id)
    orig_width, orig_height = img.size

    st.image(img, caption="Raw image from Drive", use_column_width=False)

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

    # helper to unscale
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

    # Single canvas per step
    def draw_step(step_num, instruction, mode, color, key_suffix):
        st.markdown(f"### {instruction}")
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3 if mode == "line" else 2,
            stroke_color=color,
            background_color="#000000",
            background_image=img,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=mode,
            key=f"canvas_step{step_num}_{file_id}",
            display_toolbar=True,
        )
        return canvas

    # ------- Step-specific logic ------- #

    if step == 1:
        # Clip bbox + axis
        st.write("Draw ONE rectangle around the whole clip, and ONE line along its axis.")
        canvas = draw_step(
            1,
            "Step 1: Clip bounding box (red) and axis line",
            mode="freedraw",  # we'll still parse rect+line from objects
            color="#FF0000",
            key_suffix="clip",
        )
        clip_bbox = None
        axis = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect" and clip_bbox is None:
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    clip_bbox = (left, top, left + w, top + h)
                elif obj["type"] == "line" and axis is None:
                    axis = (obj["x1"], obj["y1"], obj["x2"], obj["y2"])

        st.info(f"Clip bbox: {clip_bbox}")
        st.info(f"Axis: {axis}")

        if st.button("Next (left clip body)"):
            if clip_bbox is None or axis is None:
                st.error("Please draw BOTH a rectangle and a line.")
            else:
                raw_shapes["clip_bbox"] = clip_bbox
                raw_shapes["axis"] = axis
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 2
                st.rerun()

    elif step == 2:
        # Left clip body
        st.write("Draw ONE rectangle around the **LEFT** clip body.")
        canvas = draw_step(
            2,
            "Step 2: Left clip body (cyan)",
            mode="rect",
            color="#00FFFF",
            key_suffix="left_clip",
        )
        left_clip = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    left_clip = (left, top, left + w, top + h)
                    break

        st.info(f"Left clip bbox: {left_clip}")

        cols = st.columns(2)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 1
            st.rerun()
        if cols[1].button("Next (right clip body)"):
            if left_clip is None:
                st.error("Please draw a rectangle for the left clip.")
            else:
                raw_shapes["left_clip_bbox"] = left_clip
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 3
                st.rerun()

    elif step == 3:
        # Right clip body
        st.write("Draw ONE rectangle around the **RIGHT** clip body.")
        canvas = draw_step(
            3,
            "Step 3: Right clip body (orange)",
            mode="rect",
            color="#FFA500",
            key_suffix="right_clip",
        )
        right_clip = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    right_clip = (left, top, left + w, top + h)
                    break

        st.info(f"Right clip bbox: {right_clip}")

        cols = st.columns(2)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 2
            st.rerun()
        if cols[1].button("Next (anterior leaflet)"):
            if right_clip is None:
                st.error("Please draw a rectangle for the right clip.")
            else:
                raw_shapes["right_clip_bbox"] = right_clip
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 4
                st.rerun()

    elif step == 4:
        # Anterior leaflet
        st.write("Draw ONE rectangle around the **ANTERIOR** leaflet.")
        canvas = draw_step(
            4,
            "Step 4: Anterior leaflet (yellow)",
            mode="rect",
            color="#FFFF00",
            key_suffix="ant_leaflet",
        )
        ant = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    ant = (left, top, left + w, top + h)
                    break

        st.info(f"Anterior leaflet bbox: {ant}")

        cols = st.columns(2)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 3
            st.rerun()
        if cols[1].button("Next (posterior leaflet)"):
            if ant is None:
                st.error("Please draw a rectangle for the anterior leaflet.")
            else:
                raw_shapes["ant_leaflet_bbox"] = ant
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 5
                st.rerun()

    elif step == 5:
        # Posterior leaflet
        st.write("Draw ONE rectangle around the **POSTERIOR** leaflet.")
        canvas = draw_step(
            5,
            "Step 5: Posterior leaflet (blue)",
            mode="rect",
            color="#0000FF",
            key_suffix="post_leaflet",
        )
        post = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    post = (left, top, left + w, top + h)
                    break

        st.info(f"Posterior leaflet bbox: {post}")

        cols = st.columns(2)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 4
            st.rerun()
        if cols[1].button("Next (left clip stem)"):
            if post is None:
                st.error("Please draw a rectangle for the posterior leaflet.")
            else:
                raw_shapes["post_leaflet_bbox"] = post
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 6
                st.rerun()

    elif step == 6:
        # Left stem
        st.write("Draw ONE rectangle around the **LEFT** clip stem.")
        canvas = draw_step(
            6,
            "Step 6: Left clip stem (magenta)",
            mode="rect",
            color="#FF00FF",
            key_suffix="left_stem",
        )
        lstem = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    lstem = (left, top, left + w, top + h)
                    break

        st.info(f"Left stem bbox: {lstem}")

        cols = st.columns(2)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 5
            st.rerun()
        if cols[1].button("Next (right clip stem)"):
            if lstem is None:
                st.error("Please draw a rectangle for the left stem.")
            else:
                raw_shapes["left_stem_bbox"] = lstem
                st.session_state["raw_shapes"] = raw_shapes
                st.session_state["label_step"] = 7
                st.rerun()

    elif step == 7:
        # Right stem + submit
        st.write("Draw ONE rectangle around the **RIGHT** clip stem.")
        canvas = draw_step(
            7,
            "Step 7: Right clip stem (green)",
            mode="rect",
            color="#00FF00",
            key_suffix="right_stem",
        )
        rstem = None
        if canvas.json_data is not None:
            for obj in canvas.json_data.get("objects", []):
                if obj["type"] == "rect":
                    left = obj["left"]; top = obj["top"]
                    w = obj["width"]; h = obj["height"]
                    rstem = (left, top, left + w, top + h)
                    break

        st.info(f"Right stem bbox: {rstem}")

        cols = st.columns(2)
        if cols[0].button("Back"):
            st.session_state["label_step"] = 6
            st.rerun()

        if cols[1].button("Submit all labels for this image"):
            if rstem is None:
                st.error("Please draw a rectangle for the right stem.")
            else:
                raw_shapes["right_stem_bbox"] = rstem

                # pull everything out, unscale to original coords
                clip_bbox = raw_shapes.get("clip_bbox")
                axis = raw_shapes.get("axis")
                left_clip = raw_shapes.get("left_clip_bbox")
                right_clip = raw_shapes.get("right_clip_bbox")
                ant = raw_shapes.get("ant_leaflet_bbox")
                post = raw_shapes.get("post_leaflet_bbox")
                lstem = raw_shapes.get("left_stem_bbox")
                rstem = raw_shapes.get("right_stem_bbox")

                clip_x1, clip_y1, clip_x2, clip_y2 = unscale_bbox(clip_bbox)
                axis_x1, axis_y1, axis_x2, axis_y2 = unscale_line(axis)
                lclip_x1, lclip_y1, lclip_x2, lclip_y2 = unscale_bbox(left_clip)
                rclip_x1, rclip_y1, rclip_x2, rclip_y2 = unscale_bbox(right_clip)
                ant_x1, ant_y1, ant_x2, ant_y2 = unscale_bbox(ant)
                post_x1, post_y1, post_x2, post_y2 = unscale_bbox(post)
                lstem_x1, lstem_y1, lstem_x2, lstem_y2 = unscale_bbox(lstem)
                rstem_x1, rstem_y1, rstem_x2, rstem_y2 = unscale_bbox(rstem)

                new_row = {
                    "image_name": image_name,
                    "drive_file_id": file_id,
                    "annotator": annotator,
                    "clip_bbox_x1": clip_x1,
                    "clip_bbox_y1": clip_y1,
                    "clip_bbox_x2": clip_x2,
                    "clip_bbox_y2": clip_y2,
                    "axis_x1": axis_x1,
                    "axis_y1": axis_y1,
                    "axis_x2": axis_x2,
                    "axis_y2": axis_y2,
                    "left_clip_bbox_x1": lclip_x1,
                    "left_clip_bbox_y1": lclip_y1,
                    "left_clip_bbox_x2": lclip_x2,
                    "left_clip_bbox_y2": lclip_y2,
                    "right_clip_bbox_x1": rclip_x1,
                    "right_clip_bbox_y1": rclip_y1,
                    "right_clip_bbox_x2": rclip_x2,
                    "right_clip_bbox_y2": rclip_y2,
                    "ant_leaflet_bbox_x1": ant_x1,
                    "ant_leaflet_bbox_y1": ant_y1,
                    "ant_leaflet_bbox_x2": ant_x2,
                    "ant_leaflet_bbox_y2": ant_y2,
                    "post_leaflet_bbox_x1": post_x1,
                    "post_leaflet_bbox_y1": post_y1,
                    "post_leaflet_bbox_x2": post_x2,
                    "post_leaflet_bbox_y2": post_y2,
                    "left_stem_bbox_x1": lstem_x1,
                    "left_stem_bbox_y1": lstem_y1,
                    "left_stem_bbox_x2": lstem_x2,
                    "left_stem_bbox_y2": lstem_y2,
                    "right_stem_bbox_x1": rstem_x1,
                    "right_stem_bbox_y1": rstem_y1,
                    "right_stem_bbox_x2": rstem_x2,
                    "right_stem_bbox_y2": rstem_y2,
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

                # reset wizard for next image
                st.session_state.pop("current_file_id", None)
                st.session_state.pop("current_image_name", None)
                st.session_state.pop("raw_shapes", None)
                st.session_state["label_step"] = 1

                st.success("Labels saved! Loading next image...")
                st.rerun()
