import argparse
import os
import cv2
import numpy as np

from opencv_green_detection import (
	GreenLightDetector,
	create_trackbars,
	get_trackbar_values,
	detect_green_light_and_offset,
	get_debug_images,
)


class ImageDebugTool:
	def __init__(self) -> None:
		self.detector = GreenLightDetector()
		self.original_image: np.ndarray | None = None
		self.last_result_image: np.ndarray | None = None
		self.window_created = False

	def load_image(self, image_path: str) -> bool:
		if not os.path.exists(image_path):
			print(f"错误：图片文件不存在 - {image_path}")
			return False
		image = cv2.imread(image_path)
		if image is None:
			print(f"错误：无法读取图片 - {image_path}")
			return False
		self.original_image = image
		print(f"成功加载图片: {image_path}")
		print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
		return True

	def create_windows_and_trackbars(self) -> None:
		if self.window_created:
			return
		cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
		cv2.namedWindow('HSV Mask', cv2.WINDOW_NORMAL)
		cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
		create_trackbars()
		self.window_created = True

	def render_overlays(
		self,
		frame: np.ndarray,
		contour_info: dict | None,
		lower_hsv: np.ndarray,
		upper_hsv: np.ndarray,
	) -> np.ndarray:
		result = frame.copy()
		# 顶部左侧显示当前HSV范围
		cv2.putText(
			result,
			f"H:[{int(lower_hsv[0])}-{int(upper_hsv[0])}] S:[{int(lower_hsv[1])}-{int(upper_hsv[1])}] V:[{int(lower_hsv[2])}-{int(upper_hsv[2])}]",
			(10, 28),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)
		if contour_info is not None:
			center = contour_info.get('center')
			area = contour_info.get('area')
			circle = contour_info.get('circle')
			if circle is not None:
				x, y, r = circle
				cv2.circle(result, (x, y), r, (0, 255, 255), 2)
			if center is not None:
				cv2.circle(result, center, 5, (0, 0, 255), -1)
			cv2.putText(
				result,
				f"Area:{int(area)}",
				(10, 56),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2,
			)
		return result

	def step(self) -> bool:
		if self.original_image is None:
			return False
		lower_hsv, upper_hsv = get_trackbar_values()
		success, offset, contour_info = detect_green_light_and_offset(
			self.original_image, lower_hsv, upper_hsv
		)
		debug_mask, debug_result = get_debug_images()
		# 如果底层未提供调试图像，则自行生成mask
		if debug_mask is None:
			hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
			debug_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
		# 叠加可视化
		self.last_result_image = self.render_overlays(
			self.original_image, contour_info if success else None, lower_hsv, upper_hsv
		)
		cv2.imshow('Original Image', self.original_image)
		cv2.imshow('HSV Mask', debug_mask)
		cv2.imshow('Detection Result', self.last_result_image)
		# 控制台输出一次关键信息
		if success:
			print(f"检测到目标，偏移: {offset}，面积: {int(contour_info['area'])}")
		else:
			print("未检测到目标")
		return True

	def save(self, output_path: str) -> None:
		if self.last_result_image is None:
			print("没有可保存的结果，请先进行一次检测显示。")
			return
		cv2.imwrite(output_path, self.last_result_image)
		print(f"结果已保存到: {output_path}")

	def run(self, image_path: str) -> None:
		if not self.load_image(image_path):
			return
		self.create_windows_and_trackbars()
		print("\n操作说明: q 退出, s 保存结果, r 重新加载图片")
		while True:
			self.step()
			key = cv2.waitKey(100) & 0xFF
			if key == ord('q'):
				break
			elif key == ord('s'):
				base = os.path.basename(image_path)
				name, ext = os.path.splitext(base)
				self.save(f"{name}_debug{ext}")
			elif key == ord('r'):
				self.load_image(image_path)
		cv2.destroyAllWindows()


def main() -> None:
	parser = argparse.ArgumentParser(description='OpenCV 绿色检测 图片调试工具')
	parser.add_argument('image', help='要调试的图片路径')
	args = parser.parse_args()
	tool = ImageDebugTool()
	tool.run(args.image)


if __name__ == '__main__':
	main()


