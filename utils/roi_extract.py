from scipy.ndimage import median_filter as scipy_median_filter
import numpy as np
import cv2


class ROI:
	"""Region of Interest (ROI) class."""

	def __init__(self, image, kernel_size=3, norm=True, edge_method='sobel'):
		"""
		Khởi tạo đối tượng ROI và crop sát nhất với não bằng bounding box non-zero pixels.
		Args:
			image (np.ndarray): Ảnh MRI đầu vào
			kernel_size (int): Kích thước kernel cho median filter
			norm (bool): Có chuẩn hoá ảnh không
			edge_method (str): 'sobel' hoặc 'canny'
		"""
		self.check_grayscale(image)
		filtered = self.median_filter(image, kernel_size)
		if norm:
			normalized = self.linear_normalization(filtered)
		else:
			normalized = filtered
		# Tìm bounding box của vùng não (non-zero pixels)
		mask = normalized > 0
		if np.any(mask):
			y_indices = np.where(np.any(mask, axis=1))[0]
			x_indices = np.where(np.any(mask, axis=0))[0]
			y_min, y_max = y_indices[0], y_indices[-1]
			x_min, x_max = x_indices[0], x_indices[-1]
			roi = normalized[y_min:y_max+1, x_min:x_max+1]
			self.roi = roi
		else:
			self.roi = normalized

	def check_grayscale(self, image):
		"""
		Kiểm tra xem ảnh có phải là ảnh xám (gray scale) hay không.
		Nếu không phải, raise ValueError.
		Args:
			image (np.ndarray): Ảnh đầu vào.
		Raises:
			ValueError: Nếu ảnh không phải là ảnh xám.
		"""
		if not isinstance(image, np.ndarray):
			raise ValueError("Đầu vào phải là một numpy.ndarray")
		if image.ndim == 2:
			return True
		elif image.ndim == 3 and image.shape[2] == 1:
			return True
		else:
			raise ValueError("Ảnh không phải là ảnh xám (gray scale)")

	def median_filter(self, image, kernel_size=3):
		"""
		Lọc nhiễu bằng median filter để loại bỏ noise và outlier.
		Args:
			image (np.ndarray): Ảnh đầu vào.
			kernel_size (int): Kích thước kernel cho median filter. Mặc định là 3.
		Returns:
			Ảnh đã được lọc nhiễu.
		"""
		return scipy_median_filter(image, size=kernel_size)

	def linear_normalization(self, image):
		"""
		Chuẩn hoá ảnh xám bằng linear normalization để cân bằng độ sáng và contrast.
		Args:
			image (np.ndarray): Ảnh đầu vào.
		Returns:
			Ảnh đã được chuẩn hoá về dải giá trị [0, 255] (kiểu uint8).
		"""
		img_min = image.min()
		img_max = image.max()
		if img_max == img_min:
			return np.zeros_like(image, dtype=np.uint8)
		normalized = (image - img_min) / (img_max - img_min)
		return (normalized * 255).astype(np.uint8)

	def edge_detection(self, image, method='sobel'):
		"""
		Phát hiện biên trên ảnh xám bằng Sobel hoặc Canny.
		Args:
			image (np.ndarray): Ảnh đầu vào (đã chuẩn hoá, kiểu uint8).
			method (str): 'sobel' hoặc 'canny'.
		Returns:
			Ảnh biên (edge map).
		"""
		if method == 'sobel':
			# Sobel theo cả hai hướng
			edge_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
			edge_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
			edges = np.sqrt(edge_x**2 + edge_y**2)
			edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
			return edges
		elif method == 'canny':
			return cv2.Canny(image, 50, 150)
		else:
			raise ValueError("Chỉ hỗ trợ method 'sobel' hoặc 'canny'")
