import numpy as np
import qrcode
from PIL import Image, ImageDraw
from qrcode.image.styles.colormasks import ImageColorMask

def inRange(value: float, start: float, end: float) -> bool: # value 是否介於 start 和 end 之間
	return value >= start and value <= end

def tupleAdd(t1: tuple[float, float], t2: tuple[float, float]) -> tuple[float, float]: # 將兩個 tuple[float, float] 相加
	return (t1[0]+t2[0], t1[1]+t2[1])

def tupleMulti(t: tuple[float, float], value: float): # 將 tuple[float, float] 每個分量都乘上常數 value
	return (t[0]*value, t[1]*value)

def getQRCodeArr(s_data: str, level: int) -> list[list[int]]: # 生成 QRCode 矩陣
	qr = qrcode.QRCode(error_correction=level, box_size=1, border=1) # 設定 QRCode 參數
	qr.add_data(s_data) # 放入資料
	qr.make(fit=True) # 計算 QRCode
	image=Image.open("images/image_control_net.png")
	image = image.convert('RGB')

	PILimg_QRCode = qr.make_image(fill_color="#000", back_color="#fff") # 生成 QRCode 圖片

	#PILimg_QRCode.show()
	PILimg_QRCode.save("images/QRCode.png")

	NParr_QRCodeImg = np.array(PILimg_QRCode) # 圖片轉 np 陣列

	return [[0 if j[0] == 255 else 1 for j in i] for i in NParr_QRCodeImg],PILimg_QRCode.size # 生成 白為0 黑為1 的矩陣

def drawColorDot(draw: ImageDraw, pos: tuple[float, float], radius: float, colorCode: str) -> None: # 繪製色點
	point1 = (pos[0]-radius, pos[1]-radius)
	point2 = (pos[0]+radius, pos[1]+radius)
	draw.ellipse([point1, point2], fill="#"+colorCode)

def drawColorRect(draw: ImageDraw, pos: tuple[float, float], radius: float, colorCode: str) -> None: # 繪製色點
	point1 = (pos[0]-radius, pos[1]-radius)
	point2 = (pos[0]+radius, pos[1]+radius)
	draw.rectangle([point1, point2], fill="#"+colorCode)

def insertQRCodeToImg(
	inputImageUrl: str, # 要美化的圖片的檔案路徑
	outputImageUrl: str, #
	s_data: str,
	correctLevel: int,
	finderPattern: str,
	radiusRate: float = 1
) -> None: # 將字串轉為 QRCode 後
	img_input = Image.open(inputImageUrl).convert("RGBA") # 讀取圖片
	imageOriginalSize = img_input.size # 保存圖片的原始大小
	img_input = img_input.resize((2048, 2048), Image.LANCZOS) # 等比例放大圖片, 因為要使 QRCode 像素點平滑, 等等會縮小回原大小

	img_overlay = Image.new("RGBA", img_input.size, (255, 255, 255, 0)) # 透明圖層, 用於繪製額外的東西在原始圖片上
	draw = ImageDraw.Draw(img_overlay)

	a_QRCode,qr_size = getQRCodeArr(s_data, correctLevel) # 取得 QRCode 矩陣
	#print(a_QRCode)
	QRCodeWidth = len(a_QRCode) # QRCode 的像素邊長 (px)
	pixelWidth = img_input.size[0] / QRCodeWidth # 每個 QRCode 像素對應到要美化的圖片上, 的大小

	image = img_input

	image = image.convert('RGB')

	image=image.resize(qr_size)
	image_array = np.array(image)
	#print(len(image_array))
	#print(image_array)

	for i in range(QRCodeWidth): # 繪製 QRCode 的每個點
		for j in range(QRCodeWidth):

			r,g,b=image_array[j][i]
			'''r1,g1,b1=0,0,0
			r2,g2,b2=255,255,255'''
			r1=r-53
			g1=g-53
			b1=b-53

			r2=r+47
			g2=g+47
			b2=b+47

			if r1<0:
				r1=0
			if g1<0:
				g1=0
			if b1<0:
				b1=0

			if r2>255:
				r2=255
			if g2>255:
				g2=255
			if b2>255:
				b2=255

			color = "{:02x}{:02x}{:02x}".format(r1, g1, b1) if a_QRCode[j][i] else "{:02x}{:02x}{:02x}".format(r2, g2, b2)# QRCode 像素點是黑色還是白色

			if (i < 9 and j < 9) or (i >= QRCodeWidth-9 and j < 9) or (i < 9 and j >= QRCodeWidth-9) or\
				 (inRange(i, QRCodeWidth-10, QRCodeWidth-6) and inRange(j, QRCodeWidth-10, QRCodeWidth-6)):
				if finderPattern == "rect": drawColorRect(draw, tupleMulti((0.5+i, 0.5+j), pixelWidth), pixelWidth/2, color) # 繪製定位點
			elif inRange(i, 1, QRCodeWidth-2) and inRange(j, 1, QRCodeWidth-2):
				drawColorDot(draw, tupleMulti((0.5+i, 0.5+j), pixelWidth), pixelWidth/2 * radiusRate, color) # 繪製資料點

	if finderPattern == "circle":
		for pos in ((4.5, 4.5), (4.5, QRCodeWidth-4.5), (QRCodeWidth-4.5, 4.5)): # 繪製三個圓形的大定位點
			for pixelRadius, color in ((4.5, "fff"), (3.5, "000"), (2.5, "fff"), (1.5, "000")): # 每個大定位點都有黑白相間的圓
				drawColorDot(draw, tupleMulti(pos, pixelWidth), pixelWidth*pixelRadius, color+"b")

		for pixelRadius, color in ((2.5, "000"), (1.5, "fff"), (0.5, "000")): # 繪製右下角的小定位點
			drawColorDot(draw, tupleMulti((QRCodeWidth-7.5, QRCodeWidth-7.5), pixelWidth), pixelWidth*pixelRadius, color+"4")

	img_input = Image.alpha_composite(img_input, img_overlay) # 將繪製的 QRCode 覆蓋到要美化的圖片上
	img_input = img_input.resize(imageOriginalSize, Image.LANCZOS) # 等比例縮小圖片
	img_input.show()
	img_input.save(outputImageUrl) # 以給定路徑保存圖片

#url = "https://www.youtube.com/@ShishiroBotan"
url = "https://hackmd.io/8c_mOP2qS0u8Q9Oy82fZYQ"

insertQRCodeToImg("images/image_control_net.png", "images/output.png", url, qrcode.ERROR_CORRECT_H, "circle", 0.2)