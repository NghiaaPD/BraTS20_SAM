# BraTS2020 Dataset Documentation

## Giới thiệu

BraTS (Brain Tumor Segmentation) 2020 là bộ dữ liệu chuẩn cho bài toán phân đoạn khối u não từ ảnh MRI đa phương thức (multi-modal MRI).

## Cấu trúc dữ liệu

Mỗi case/bệnh nhân trong dataset bao gồm 5 file NIfTI (.nii):

Có tổng cổng **369 bệnh nhân cho tập train** và **125 bệnh nhân cho tập validation**

### 1. File ảnh MRI (4 modalities)

#### `*_flair.nii` - FLAIR (Fluid Attenuated Inversion Recovery)

- **Mục đích**: Phát hiện tổn thương và phù não
- **Đặc điểm**: Tín hiệu nước bị triệt, làm nổi bật các vùng bất thường
- **Ứng dụng**: Hiển thị rõ vùng phù não (edema) xung quanh khối u
- **Kích thước**: ~18MB

#### `*_t1.nii` - T1-weighted

- **Mục đích**: Cung cấp thông tin giải phẫu cấu trúc não
- **Đặc điểm**: Độ tương phản tốt giữa chất xám và chất trắng
- **Ứng dụng**: Xác định ranh giới giải phẫu và cấu trúc não cơ bản
- **Kích thước**: ~18MB

#### `*_t1ce.nii` - T1 Contrast-Enhanced

- **Mục đích**: Hiển thị vùng khối u có tăng cường tương phản
- **Đặc điểm**: Sử dụng chất tương phản (contrast agent) như Gadolinium
- **Ứng dụng**: Phát hiện vùng u đang hoạt động/tăng trưởng (enhancing tumor)
- **Kích thước**: ~18MB

#### `*_t2.nii` - T2-weighted

- **Mục đích**: Hiển thị vùng phù não và khối u
- **Đặc điểm**: Tín hiệu nước cao, làm sáng các vùng chứa nhiều nước
- **Ứng dụng**: Xác định ranh giới khối u và vùng phù
- **Kích thước**: ~18MB

### 2. File nhãn (Ground Truth)

#### `*_seg.nii` - Segmentation Mask

- **Mục đích**: Nhãn phân đoạn thủ công bởi chuyên gia y tế
- **Kích thước**: ~8.6MB (nhỏ hơn vì chỉ chứa giá trị integer)

**Giá trị nhãn:**

- **0**: Background (nền - mô não bình thường)
- **1**: NCR (Necrotic and Non-enhancing Tumor Core) - lõi u hoại tử/không tăng cường
- **2**: ED (Peritumoral Edema) - phù não quanh u
- **4**: ET (Enhancing Tumor) - khối u tăng cường

**Nhóm vùng quan trọng:**

- **Whole Tumor (WT)**: Toàn bộ khối u = labels {1, 2, 4}
- **Tumor Core (TC)**: Lõi u = labels {1, 4}
- **Enhancing Tumor (ET)**: U tăng cường = label {4}

## Định dạng file

- **Format**: NIfTI (Neuroimaging Informatics Technology Initiative)
- **Extension**: `.nii` hoặc `.nii.gz` (compressed)
- **Dimensions**: Thường là 3D volume (240 x 240 x 155)
- **Voxel spacing**: 1mm x 1mm x 1mm (isotropic)
- **Data type**:
  - Ảnh MRI: float32 hoặc int16
  - Segmentation: uint8

## Chi tiết

**Shape: (240, 240, 155)**:

- 240 voxel theo trục X

- 240 voxel theo trục Y

- 155 lát theo trục Z

```
   ↑ Y (240 pixels)
   |
   |
   +--------→X (240 pixels)
  /
 /
/  Z (155 slices)
```

**Data type: float64**\
**Min value: 0.00**\
**Max value: 625.00**\
**Mean value: 26.02**

- Min = 0
- Max = 625
- Mean = 26.02

→ Đây là intensity MRI (_Không phải HU như CT, mà là giá trị cường độ MRI_)

**Voxel dimensions: (np.float32(1.0), np.float32(1.0), np.float32(1.0))**

Điều này nghĩa là:

- 1 voxel = 1mm × 1mm × 1mm

→ Volume MRI có kích thước thật:

```
X: 240 mm x 1mm
Y: 240 mm x 1mm
Z: 155 mm x 1mm
```

→ Bộ não trong ảnh này xấp xỉ:

24 cm × 24 cm × 15.5 cm (Đây là kích thước không đúng do bị vướng các vùng đen bao quanh não )

**Segmentation labels**

- 0: Background
- 1: NCR (Necrotic/Non-Enhancing Core)
- 2: Edema
- 4: Enhancing Tumor

## Ứng dụng

1. **Training Deep Learning Models**: U-Net, V-Net, nnU-Net cho semantic segmentation
2. **Medical Image Analysis**: Nghiên cứu đặc điểm khối u não
3. **Computer-Aided Diagnosis**: Hỗ trợ bác sĩ trong chẩn đoán
4. **Treatment Planning**: Lập kế hoạch xạ trị, phẫu thuật

## Tham khảo

- **Dataset**: https://www.med.upenn.edu/cbica/brats2020/
- **Paper**: BraTS 2020 Challenge (MICCAI 2020)
- **Evaluation Metrics**: Dice Score, Hausdorff Distance, Sensitivity, Specificity
