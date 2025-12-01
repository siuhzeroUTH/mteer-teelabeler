import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
import json

from PIL import Image
from streamlit_drawable_canvas import st_canvas

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# ----------------- CONFIG & GOOGLE DRIVE HELPERS ----------------- #

# Folder IDs and labels filename from Streamlit secrets
DRIVE_UNLABELED_ID = st.secrets["drive"]["unlabeled_folder_id"]
DRIVE_LABELED_ID   = st.secrets["drive"]["labeled_folder_id"]
DRIVE_META_ID      = st.secrets["drive"]["meta_folder_id"]
LABELS_FILENAME    = st.secrets["drive"]["labels_filename"]

@st.cache_resource
def get_drive_service():
    # st.secrets["service_account"] is already a dict-like object from TOML
    info = dict(st.secrets["service_account"])
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)
    return service

def list_unlabeled_images(service, page_size=1):
    """Return a list of image files (id, name) in the unlabeled folder."""
    query = f"'{DRIVE_UNLABELED_ID}' in parents and mimeType contains 'image/' and trashed = false"
    r = service.files().list(
        q=query,
        pageSize=page_size,
        fields="files(id, name)"
    ).execute()
    return r.get("files", [])

def download_image_as_pil(service, file_id):
    """Download an image file from Drive and return as a PIL Image."""
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    img = Image.open(fh).convert("RGB")
    return img

def move_file_to_labeled(service, file_id):
    """Move file from unlabeled folder to labeled folder in Drive."""
    file = service.files().get(fileId=file_id, fields="parents").execute()
    prev_parents = ",".join(file.get("parents", []))
    service.files().update(
        fileId=file_id,
        addParents=DRIVE_LABELED_ID,
        removeParents=prev_parents,
        fields="id, parents"
    ).execute()

def get_or_create_labels_df(service):
    """Load labels.csv from Drive meta folder, or create it if missing."""
    query = f"'{DRIVE_META_ID}' in parents and name = '{LABELS_FILENAME}' and trashed = false"
    r = service.files().list(q=query, fields="files(id)").execute()
    files = r.get("files", [])

    if files:
        file_id = files[0]["id"]
        req = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        if fh.getbuffer().nbytes > 0:
            df = pd.read_csv(fh)
        else:
            df = pd.DataFrame(columns=[
                "image_name","drive_file_id","annotator",
                "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                "axis_x1","axis_y1","axis_x2","axis_y2",
                "created_at"
            ])
    else:
        df = pd.DataFrame(columns=[
            "image_name","drive_file_id","annotator",
            "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
            "axis_x1","axis_y1","axis_x2","axis_y2",
            "created_at"
        ])
        fh = BytesIO()
        df.to_csv(fh, index=False)
        fh.seek(0)
        meta = {
            "name": LABELS_FILENAME,
            "parents": [DRIVE_META_ID],
            "mimeType": "text/csv"
        }
        body = MediaIoBaseUpload(fh, mimetype="text/csv")
        file = service.files().create(
            body=meta,
            media_body=body,
            fields="id"
        ).execute()
        file_id = file["id"]

    return file_id, df

def save_labels_df(service, file_id, df):
    """Overwrite labels.csv in Drive with the updated DataFrame."""
    fh = BytesIO()
    df.to_csv(fh, index=False)
    fh.seek(0)
    body = MediaIoBaseUpload(fh, mimetype="text/csv")
    service.files().update(
        fileId=file_id,
        media_body=body
    ).execute()

# ----------------- STREAMLIT APP LAYOUT ----------------- #

st.set_page_config(page_title="TEE Clip Labeling", layout="wide")

# Initialize Drive service and labels DataFrame
service = get_drive_service()
labels_file_id, labels_df = get_or_create_labels_df(service)

# Sidebar: annotator identity and page selection
st.sidebar.title("Annotator")
annotator = st.sidebar.text_input("Enter your ID/name", value="", max_chars=50)
if not annotator:
    st.sidebar.warning("Please enter your annotator ID to start labeling.")

page = st.sidebar.radio("Page", ["Dashboard", "Labeling"])

# ----------------- DASHBOARD PAGE ----------------- #

if page == "Dashboard":
    st.title("Labeling Dashboard")

    # Count unlabeled images in Drive
    unlabeled_files = list_unlabeled_images(service, page_size=1000)
    total_unlabeled = len(unlabeled_files)
    total_labeled = len(labels_df)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Unlabeled images", total_unlabeled)
    with c2:
        st.metric("Labeled images", total_labeled)

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

    # Get next unlabeled image from Drive
    unlabeled_files = list_unlabeled_images(service, page_size=1)
    if not unlabeled_files:
        st.success("No more unlabeled images in Drive folder.")
        st.stop()

    current_file = unlabeled_files[0]
    file_id = current_file["id"]
    image_name = current_file["name"]

    st.subheader(f"Current image: {image_name}")

    # Download and show image
    img = download_image_as_pil(service, file_id)
    width, height = img.size

    # If image is huge, optionally downscale for display (coordinates will be in displayed space)
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
        background_color="#e6e6e6",
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
            # Save coordinates in original image space (undo scaling if we resized)
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

            move_file_to_labeled(service, file_id)

            st.success("Label saved! Loading next image...")
            st.rerun()
