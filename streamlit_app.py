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

    # Create a simple blank image so we have something to draw on
    width, height = 512, 512
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # optional: draw a mock "clip" diagonal line in the center
    draw.line((150, 150, 360, 360), fill=(0, 0, 0), width=3)

    st.write(f"Image size: {width} x {height}")
    st.write("Draw ONE bounding box around the 'clip', then ONE line along its length (tip-to-tip).")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=img,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="transform",
        key="canvas",
        display_toolbar=True,
    )

    bbox_coords = None
    axis_coords = None

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        for obj in objects:
            otype = obj.get("type", "")
            if otype == "rect" and bbox_coords is None:
                left = obj["left"]
                top = obj["top"]
                w = obj["width"]
                h = obj["height"]
                bbox_coords = (left, top, left + w, top + h)
            elif otype == "line" and axis_coords is None:
                x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
                axis_coords = (x1, y1, x2, y2)

    st.write("Parsed bounding box:", bbox_coords)
    st.write("Parsed axis line:", axis_coords)

    if st.button("Submit label"):
        if bbox_coords is None or axis_coords is None:
            st.error("Please draw ONE bounding box and ONE line before submitting.")
        else:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_coords
            axis_x1, axis_y1, axis_x2, axis_y2 = axis_coords

            new_row = {
                "image_name": image_name,
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

            st.session_state.labels_df = pd.concat(
                [st.session_state.labels_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            st.success("Label saved for this session. Moving to next fake image...")
            st.session_state.image_idx += 1
            st.experimental_rerun()
