import numpy as np
import qrcode
from PIL import Image, ImageDraw,ImageEnhance,ImageStat
from qrcode.image.styles.colormasks import ImageColorMask

img=Image.open("images\\test_0807.jpg")
rgb_img=img.convert('RGB')
L_img=img.convert('L')
pixel=rgb_img.load()
w,h=img.size
print(w,h)
stat=ImageStat.Stat(L_img)
bright_mean=stat.mean[0]
print(bright_mean)

img_dot = Image.new("RGBA", img.size, (255, 0, 0, 255)) # 透明圖層, 用於繪製額外的東西在原始圖片上
#img_dot.show()
pixel_dot=img_dot.load()
img_input=rgb_img.convert('RGBA')
pixel=img_input.load()

draw = ImageDraw.Draw(img_dot)
for i in range(0,w,10):
    for j in range(0,h,10):
        black,white,gray=0,0,0
        end_w=i+10
        end_h=j+10
        if end_w>=w:
            end_w=w-1
        if end_h>=h:
            end_h=h-1
        for x in range(i,end_w):
            for y in range(j,end_h):
                r,g,b,a=pixel[x,y]
                bright=0.299*r+0.587*g+0.114*b
                if bright>=bright_mean*1.2:
                    white+=1
                elif bright<bright_mean*0.8:
                    black+=1
        px=(i+end_w)//2
        py=(j+end_h)//2
        if px>=w:
            px=(w-1+i)//2
        if py>=h:
            py=(h-1+j)//2
        if black==100:
            draw.ellipse((px-1,py-1,px+1,py+1),'black')
        elif white==100:
            draw.ellipse((px-1,py-1,px+1,py+1),'white')
        else:
            draw.ellipse((px-1,py-1,px+1,py+1),'gray')
            
        
#print(img_dot.size)
img_dot.show()
img_dot.save("images/image_dot.png")