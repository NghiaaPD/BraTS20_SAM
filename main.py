import streamlit as st
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import SimpleITK as sitk
import os
import tempfile

def normalize_to_uint8(img):
    img = np.nan_to_num(img)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = img * 0
    img = (img * 255).astype(np.uint8)
    return img

def load_medical_image(uploaded_file):
    """Load .nii or .mha file from uploaded file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.nii', '.nii.gz']:
            img = nib.load(tmp_path)
            data = img.get_fdata()
        elif file_extension == '.mha':
            img = sitk.ReadImage(tmp_path)
            data = sitk.GetArrayFromImage(img)
        else:
            raise ValueError("Unsupported file format. Please upload .nii, .nii.gz, or .mha files.")
    finally:
        os.unlink(tmp_path)
    
    return data

st.title("Medical Image Visualization App")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    flair_file = st.file_uploader("Upload FLAIR Image (.nii, .nii.gz, .mha)", type=['nii', 'nii.gz', 'mha'])
with col2:
    seg_file = st.file_uploader("Upload Segmentation Label (.nii, .nii.gz, .mha) - Optional", type=['nii', 'nii.gz', 'mha'])

flair_data = None
seg_data = None

if flair_file:
    try:
        flair_data = load_medical_image(flair_file)
        st.success(f"Loaded FLAIR: Shape {flair_data.shape}")
    except Exception as e:
        st.error(f"Error loading FLAIR: {e}")

if seg_file:
    try:
        seg_data = load_medical_image(seg_file)
        st.success(f"Loaded Segmentation: Shape {seg_data.shape}")
    except Exception as e:
        st.error(f"Error loading Segmentation: {e}")

if flair_data is None:
    st.warning("Please upload a FLAIR image to proceed.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Brain ROI Visualization", "Tumor Visualization & Measurements"])

with tab1:
    st.header("Brain ROI Visualization")
    
    # Crop bounding box for brain
    non_zero_mask = flair_data > 0
    x_indices = np.where(np.any(non_zero_mask, axis=(1, 2)))[0]
    y_indices = np.where(np.any(non_zero_mask, axis=(0, 2)))[0]
    z_indices = np.where(np.any(non_zero_mask, axis=(0, 1)))[0]

    if len(x_indices) == 0 or len(y_indices) == 0 or len(z_indices) == 0:
        st.error("No brain region found in FLAIR data!")
        st.stop()

    x_min, x_max = x_indices[0], x_indices[-1]
    y_min, y_max = y_indices[0], y_indices[-1]
    z_min, z_max = z_indices[0], z_indices[-1]
    flair_cropped = flair_data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    st.write(f"Cropped FLAIR Shape: {flair_cropped.shape}")

    # 3D coordinates for brain
    brain_mask = flair_cropped > 0
    coords = np.column_stack(np.where(brain_mask))
    st.write(f"Number of brain points after crop: {coords.shape[0]}")

    if coords.shape[0] == 0:
        st.warning("No brain points found after cropping. Using original data.")
        brain_mask = flair_data > 0
        coords = np.column_stack(np.where(brain_mask))
        st.write(f"Total brain points: {coords.shape[0]}")

    if coords.shape[0] == 0:
        st.error("No brain points found in data!")
        st.stop()

    # Downsample if too many points
    max_points = 100000
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]

    # Bounding box
    x_min_bb, y_min_bb, z_min_bb = coords.min(axis=0)
    x_max_bb, y_max_bb, z_max_bb = coords.max(axis=0)
    st.write(f"Bounding box size: {x_max_bb - x_min_bb + 1} x {y_max_bb - y_min_bb + 1} x {z_max_bb - z_min_bb + 1}")

    corners = [
        [x_min_bb, y_min_bb, z_min_bb],
        [x_max_bb, y_min_bb, z_min_bb],
        [x_max_bb, y_max_bb, z_min_bb],
        [x_min_bb, y_max_bb, z_min_bb],
        [x_min_bb, y_min_bb, z_max_bb],
        [x_max_bb, y_min_bb, z_max_bb],
        [x_max_bb, y_max_bb, z_max_bb],
        [x_min_bb, y_max_bb, z_max_bb]
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=coords[:, 2],
            colorscale='Gray',
            opacity=0.7
        )
    )])

    # Add bounding box lines
    for edge in edges:
        c1, c2 = corners[edge[0]], corners[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[c1[0], c2[0]],
            y=[c1[1], c2[1]],
            z=[c1[2], c2[2]],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (Left-Right)',
            yaxis_title='Y (Anterior-Posterior)',
            zaxis_title='Z (Superior-Inferior)'
        ),
        title='3D Brain ROI Visualization with Bounding Box',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Tumor Visualization & Measurements")
    
    if seg_data is None:
        st.warning("Please upload a segmentation file to visualize tumor and calculate measurements.")
        st.stop()
    
    # Assume voxel spacing is 1mm for simplicity (can be improved)
    voxel_spacing = [1.0, 1.0, 1.0]  # mm
    
    # Tumor mask (whole tumor: labels 1,2,4)
    tumor_mask = (seg_data == 1) | (seg_data == 2) | (seg_data == 4)
    tumor_coords = np.column_stack(np.where(tumor_mask))
    
    if tumor_coords.shape[0] == 0:
        st.error("No tumor found in segmentation data!")
        st.stop()
    
    st.write(f"Number of tumor voxels: {tumor_coords.shape[0]}")
    
    # Downsample for visualization
    max_points = 50000
    if tumor_coords.shape[0] > max_points:
        idx = np.random.choice(tumor_coords.shape[0], max_points, replace=False)
        tumor_coords = tumor_coords[idx]
    
    # Calculate tumor dimensions
    x_indices_t = np.where(np.any(tumor_mask, axis=(1, 2)))[0]
    y_indices_t = np.where(np.any(tumor_mask, axis=(0, 2)))[0]
    z_indices_t = np.where(np.any(tumor_mask, axis=(0, 1)))[0]
    
    if len(x_indices_t) > 0 and len(y_indices_t) > 0 and len(z_indices_t) > 0:
        x_min_t, x_max_t = x_indices_t[0], x_indices_t[-1]
        y_min_t, y_max_t = y_indices_t[0], y_indices_t[-1]
        z_min_t, z_max_t = z_indices_t[0], z_indices_t[-1]
        
        tumor_width = (x_max_t - x_min_t + 1) * voxel_spacing[0]
        tumor_height = (y_max_t - y_min_t + 1) * voxel_spacing[1]
        tumor_depth = (z_max_t - z_min_t + 1) * voxel_spacing[2]
        
        st.write("### Tumor Dimensions (Whole Tumor):")
        st.write(f"**Width (X-axis)**: {tumor_width:.1f} mm ({tumor_width/10:.1f} cm)")
        st.write(f"**Length (Y-axis)**: {tumor_height:.1f} mm ({tumor_height/10:.1f} cm)")
        st.write(f"**Height (Z-axis)**: {tumor_depth:.1f} mm ({tumor_depth/10:.1f} cm)")
        
        # Volume calculation
        voxel_volume = 1.0  # mm³ (assuming 1mm³)
        whole_tumor_voxels = np.sum(tumor_mask)
        whole_tumor_volume = whole_tumor_voxels * voxel_volume / 1000  # cm³
        st.write(f"**Volume**: {whole_tumor_volume:.1f} cm³ ({whole_tumor_voxels:,} voxels)")
        
        # Bounding box for tumor
        tumor_corners = [
            [x_min_t, y_min_t, z_min_t],
            [x_max_t, y_min_t, z_min_t],
            [x_max_t, y_max_t, z_min_t],
            [x_min_t, y_max_t, z_min_t],
            [x_min_t, y_min_t, z_max_t],
            [x_max_t, y_min_t, z_max_t],
            [x_max_t, y_max_t, z_max_t],
            [x_min_t, y_max_t, z_max_t]
        ]
        tumor_edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        fig_tumor = go.Figure(data=[go.Scatter3d(
            x=tumor_coords[:, 0],
            y=tumor_coords[:, 1],
            z=tumor_coords[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.8
            ),
            name='Tumor'
        )])
        
        # Add tumor bounding box
        for edge in tumor_edges:
            c1, c2 = tumor_corners[edge[0]], tumor_corners[edge[1]]
            fig_tumor.add_trace(go.Scatter3d(
                x=[c1[0], c2[0]],
                y=[c1[1], c2[1]],
                z=[c1[2], c2[2]],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False
            ))
        
        fig_tumor.update_layout(
            scene=dict(
                xaxis_title='X (Left-Right)',
                yaxis_title='Y (Anterior-Posterior)',
                zaxis_title='Z (Superior-Inferior)'
            ),
            title='3D Tumor Visualization with Bounding Box',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig_tumor, use_container_width=True)
    else:
        st.error("Unable to calculate tumor dimensions.")