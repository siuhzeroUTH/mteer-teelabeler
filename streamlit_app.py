import os
import json
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError
import ssl

# ----------------- CONFIG FROM ENV VARIABLES ----------------- #

DRIVE_UNLABELED_ID = os.environ["DRIVE_UNLABELED_ID"]
DRIVE_LABELED_ID   = os.environ["DRIVE_LABELED_ID"]
DRIVE_META_ID      = os.environ["DRIVE_META_ID"]
LABELS_FILENAME    = os.environ.get("LABELS_FILENAME", "labels.csv")

SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]

@st.cache_resource
def get_drive_service():
    """Authenticate with Google Drive using service account JSON from env."""
    info = json.loads(SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)
    return service

def _drive_guard(fn, what: str):
    """Run a Drive API call and show a friendly error if it fails."""
    try:
        return fn()
    except (HttpError, ssl.SSLError) as e:
        st.error(
            f"Error talking to Google Drive while **{what}**.\n\n"
            "This is a network/SSL issue between the host and Google. "
            "Please refresh the page or try again later.\n\n"
            f"Details: {e}"
        )
        st.stop()

def list_unlabeled_images(service, page_size=1000):
    """Return all image files (id, name) in the unlabeled folder."""
    def _call():
        query = (
            f"'{DRIVE_UNLABELED_ID}' in parents "
            f"and mimeType contains 'image/' and trashed = false"
        )
        return service.files().list(
            q=query,
            pageSize=page_size,
            fields="files(id, name)"
        ).execute()

    r = _drive_guard(_call, "listing unlabeled images")
    return r.get("files", [])

def get_next_unlabeled_image(service, labels_df):
    """
    Pick the first image in the unlabeled folder whose drive_file_id
    is NOT already in labels_df (i.e., not already labeled).
    """
    files = list_unlabeled_images(service, page_size=1000)
    labeled_ids = set(labels_df["drive_file_id"].astype(str)) if not labels_df.empty else set()

    for f in files:
        if f["id"] not in labeled_ids:
            return f
    return None  # nothing left to label

def download_image_as_pil(service, file_id):
    """Download an image file from Drive and return as a PIL Image."""
    def _call():
        request = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh

    fh = _drive_guard(_call, f"downloading image file {file_id}")
    img = Image.open(fh).convert("RGB")
    return img

def move_file_to_labeled(service, file_id):
    """
    Try to move file from unlabeled folder to labeled folder in Drive.
    If it fails, log a warning but DO NOT stop the app; the labeling
    queue is driven by labels.csv, not by folder membership.
    """
    try:
        file = service.files().get(fileId=file_id, fields="parents").execute()
        prev_parents = ",".join(file.get("parents", []))
        service.files().update(
            fileId=file_id,
            addParents=DRIVE_LABELED_ID,
            removeParents=prev_parents,
            fields="id, parents"
        ).execute()
    except Exception as e:
        st.warning(f"Could not move file {file_id} to labeled folder: {e}")

def get_labels_df(service):
    """
    Load labels.csv from Drive meta folder.
    We assume you manually created labels.csv in the meta folder.
    """
    def _list():
        query = f"'{DRIVE_META_ID}' in parents and trashed = false"
        return service.files().list(q=query, fields="files(id, name)").execute()

    r = _drive_guard(_list, "listing files in meta folder")
    files = r.get("files", [])

    labels_files = [f for f in files if f["name"] == LABELS_FILENAME]

    if not labels_files:
        st.error(
            f"Could not find {LABELS_FILENAME} in the meta folder.\n\n"
            f"Please create a file named '{LABELS_FILENAME}' in the meta folder "
            f"with this header row:\n\n"
            "image_name,drive_file_id,annotator,bbox_x1,bbox_y1,bbox_x2,bbox_y2,"
            "axis_x1,axis_y1,axis_x2,axis_y2,created_at"
        )
        st.stop()

    file_id = labels_files[0]["id"]

    def _get_media():
        req = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh

    fh = _drive_guard(_get_media, f"downloading {LABELS_FILENAME}")

    if fh.getbuffer().nbytes > 0:
        df = pd.read_csv(fh)
    else:
        df = pd.DataFrame(columns=[
            "image_name","drive_file_id","annotator",
            "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
            "axis_x1","axis_y1","axis_x2","axis_y2",
            "created_at"
        ])

    return file_id, df

def save_labels_df(service, file_id, df):
    """Overwrite labels.csv in Drive with the updated DataFrame."""
    def _update():
        fh = BytesIO()
        df.to_csv(fh, index=False)
        fh.seek(0)
        body = MediaIoBaseUpload(fh, mimetype="text/csv")
        return service.files().update(
            fileId=file_id,
            media_body=body
        ).execute()

    _drive_guard(_update, f"updating {LABELS_FILENAME}")

# ----------------- STREAMLIT APP LAYOUT ----------------- #

st.set_page_config(page_title="TEE Clip Labeling", layout="wide")

service = get_drive_service()
labels_file_id, labels_df = get_labels_df(service)

st.sidebar.title("Annotator")
annotator = st.sidebar.text_input("Enter your ID/name", value="", max_chars=50)
if not annotator:
    st.sidebar.warning("Please enter your annotator ID to start labeling.")

page = st.sidebar.radio("Page", ["Dashboard", "Labeling"])

# ----------------- DASHBOARD PAGE ----------------- #

if page == "Dashboard":
    st.title("Labeling Dashboard")

    all_unlabeled_files = list_unlabeled_images(service, page_size=1000)
    # Still in folder, whether labeled or not
    total_in_folder = len(all_unlabeled_files)
    # Truly unlabeled (not in labels.csv)
    labeled_ids = set(labels_df["drive_file_id"].astype(str)) if not labels_df.empty else set()
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

# ----------------- LABELING PAGE ----------------- #

if page == "Labeling":
    st.title("TEE Clip Labeling")

    if not annotator:
        st.warning("Please enter your annotator ID in the sidebar.")
        st.stop()

    current_file = get_next_unlabeled_image(service, labels_df)

    if current_file is None:
        st.success("No more unlabeled images left to label in this folder.")
        st.stop()

    file_id = current_file["id"]
    image_name = current_file["name"]

    st.subheader(f"Current image: {image_name}")
    st.write("DEBUG – current Drive file:", current_file)

    img = download_image_as_pil(service, file_id)
    width, height = img.size

    # Show raw image
    st.image(img, caption="Raw image from Drive", use_column_width=False)

    max_dim = 900
    scale = 1.0
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height))
        width, height = new_width, new_height

    st.write(f"Displayed image size: {width} x {height}")
    st.write(
        "1) Choose **Rectangle (bbox)** to draw a box around the clip.\n"
        "2) Choose **Line (axis)** to draw a line along the clip's length.\n"
        "Please draw exactly ONE rectangle and ONE line."
    )

    tool = st.radio(
        "Choose drawing tool:",
        ["Rectangle (bbox)", "Line (axis)"],
        horizontal=True,
    )
    draw_mode = "rect" if tool.startswith("Rectangle") else "line"

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_color="#000000",
        background_image=img,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=draw_mode,
        key="canvas",
        display_toolbar=True,
    )

    bbox_coords = None
    axis_coords = None

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        for obj in objects:
            if obj["type"] == "rect" and bbox_coords is None:
                left = obj["left"]
                top = obj["top"]
                w = obj["width"]
                h = obj["height"]
                bbox_coords = (left, top, left + w, top + h)

            elif obj["type"] == "line" and axis_coords is None:
                axis_coords = (obj["x1"], obj["y1"], obj["x2"], obj["y2"])

    st.info(f"Parsed bbox: {bbox_coords}")
    st.info(f"Parsed axis: {axis_coords}")

    if st.button("Submit label"):
        if bbox_coords is None or axis_coords is None:
            st.error("Please draw a rectangle AND a line before submitting.")
        else:
            if scale != 1.0:
                inv = 1.0 / scale
                bbox_x1 = bbox_coords[0] * inv
                bbox_y1 = bbox_coords[1] * inv
                bbox_x2 = bbox_coords[2] * inv
                bbox_y2 = bbox_coords[3] * inv
                axis_x1 = axis_coords[0] * inv
                axis_y1 = axis_coords[1] * inv
                axis_x2 = axis_coords[2] * inv
                axis_y2 = axis_coords[3] * inv
            else:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_coords
                axis_x1, axis_y1, axis_x2, axis_y2 = axis_coords

            new_row = {
                "image_name": image_name,
                "drive_file_id": file_id,
                "annotator": annotator,
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
                "axis_x1": axis_x1,
                "axis_y1": axis_y1,
                "axis_x2": axis_x2,
                "axis_y2": axis_y2,
                "created_at": datetime.utcnow().isoformat()
            }

            labels_df = pd.concat([labels_df, pd.DataFrame([new_row])], ignore_index=True)
            save_labels_df(service, labels_file_id, labels_df)

            # Try to move for organization, but do not rely on it for queue logic
            move_file_to_labeled(service, file_id)

            st.success("Label saved! Loading next image...")
            st.rerun()
