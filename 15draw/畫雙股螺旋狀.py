import turtle

# 初始化
screen = turtle.Screen()
screen.bgcolor("white")
screen.title("Double Spiral")

# 创建两个画笔，分别用于绘制两个螺旋线
pen1 = turtle.Turtle()
pen1.speed(0)
pen1.color("blue")

pen2 = turtle.Turtle()
pen2.speed(0)
pen2.color("red")

# 绘制双螺旋线
for i in range(360):
    pen1.forward(i)
    pen1.left(91)
    pen2.forward(i)
    pen2.right(91)

# 等待点击关闭窗口
screen.mainloop()


