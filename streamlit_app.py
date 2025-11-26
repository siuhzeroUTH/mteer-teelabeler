import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="TEE Clip Labeling (Cloud Prototype)", layout="wide")

# ---- Fake in-memory "unlabeled images" list for now ----
# Later we'll replace this with Google Drive.
FAKE_IMAGES = ["fake_case001_bicomm_frame0001.png",
               "fake_case001_bicomm_frame0002.png"]

# Session state index to walk through fake images
if "image_idx" not in st.session_state:
    st.session_state.image_idx = 0

# Sidebar: annotator ID
st.sidebar.title("Annotator")
annotator = st.sidebar.text_input("Enter your ID/name", value="", max_chars=50)
if not annotator:
    st.sidebar.warning("Please enter your annotator ID to start labeling.")

page = st.sidebar.radio("Page", ["Dashboard", "Labeling"])

# In-memory labels table (per session only, for now)
if "labels_df" not in st.session_state:
    st.session_state.labels_df = pd.DataFrame(columns=[
        "image_name", "annotator",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "axis_x1", "axis_y1", "axis_x2", "axis_y2",
        "created_at"
    ])

labels_df = st.session_state.labels_df

# -------- DASHBOARD --------
if page == "Dashboard":
    st.title("Labeling Dashboard (Cloud Prototype)")

    total_unlabeled = max(0, len(FAKE_IMAGES) - st.session_state.image_idx)
    total_labeled = len(labels_df)

    st.metric("Unlabeled images (this session)", total_unlabeled)
    st.metric("Labeled images (this session)", total_labeled)

    if total_labeled > 0:
        st.subheader("Labels per annotator")
        annot_counts = labels_df["annotator"].value_counts().reset_index()
        annot_counts.columns = ["annotator", "count"]
        st.table(annot_counts)

        st.subheader("Recent labels")
        st.dataframe(labels_df.sort_values("created_at", ascending=False).head(10))

# -------- LABELING --------
if page == "Labeling":
    st.title("TEE Clip Labeling (Cloud Prototype)")

    if not annotator:
        st.warning("Please enter your annotator ID in the sidebar.")
        st.stop()

    if st.session_state.image_idx >= len(FAKE_IMAGES):
        st.success("No more unlabeled images in this demo.")
        st.stop()

    image_name = FAKE_IMAGES[st.session_state.image_idx]
    st.subheader(f"Current image: {image_name}")

    # Create a simple background image (Pillow image)
    width, height = 512, 512
    img = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    # draw a mock "clip" diagonal
    draw.line((150, 150, 360, 360), fill=(0, 0, 0), width=4)

    st.write("Use the selector below to draw a bounding box and a line:")

    tool = st.radio(
        "Choose drawing tool:",
        ["Rectangle (bbox)", "Line (axis)"],
        horizontal=True,
    )
    draw_mode = "rect" if tool.startswith("Rectangle") else "line"

    # NOTE: background_image expects a PIL Image, NOT a numpy array
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_color="#e6e6e6",  # fallback color
        background_image=img,        # <-- pass PIL image directly
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
            new_row = {
                "image_name": image_name,
                "annotator": annotator,
                "bbox_x1": bbox_coords[0],
                "bbox_y1": bbox_coords[1],
                "bbox_x2": bbox_coords[2],
                "bbox_y2": bbox_coords[3],
                "axis_x1": axis_coords[0],
                "axis_y1": axis_coords[1],
                "axis_x2": axis_coords[2],
                "axis_y2": axis_coords[3],
                "created_at": datetime.utcnow().isoformat(),
            }

            st.session_state.labels_df = pd.concat(
                [st.session_state.labels_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )

            st.success("Label saved! Loading next image...")
            st.session_state.image_idx += 1
            st.experimental_rerun()

