import tkinter as tk

from PIL import Image, ImageDraw


class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 28
        self.sizey = 28
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.coords= []
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Done!",width=10,bg='white',command=self.save)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear!",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+20)

        # root = tk.Tk()
        # tk.Button(root, text="Quit", command=lambda root=root:quit(root)).pack()
        # root.mainloop()



        self.image=Image.new("RGB",(28,28),(0,0,0))
        # self.image = self.image.resize((28, 28))
        self.draw=ImageDraw.Draw(self.image)

    def save(self):
        print(self.coords)
        self.draw.line(self.coords,(255,255,255),width=3)
        filename = "temp.jpg"
        self.image.save(filename)

    def clear(self):
        self.drawing_area.delete("all")
        self.coords=[]

    def quit(root):
        root.destroy()

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth='true',width=3,fill='white')
                self.coords.append((self.xold,self.yold))

        self.xold = event.x
        self.yold = event.y

# if __name__ == "__main__":
#     root=tk.Tk()
#     root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
#     root.config(bg='white')
#     ImageGenerator(root,10,10)
#     root.mainloop()