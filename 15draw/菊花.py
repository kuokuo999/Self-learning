import math
import turtle

# Setup turtle environment
turtle.bgcolor("black")
turtle.pencolor("black")
turtle.shape("triangle")
turtle.speed(0)
turtle.fillcolor("orangered")

phi = 137.508 * (math.pi / 180.0)

# Loop to draw the shape
for i in range(180 + 40):
    r = 4 * math.sqrt(i)  # Calculate radius
    theta = i * phi  # Calculate angle in radians
    x = r * math.cos(theta)  # Convert polar to Cartesian x-coordinate
    y = r * math.sin(theta)  # Convert polar to Cartesian y-coordinate
    
    turtle.penup()
    turtle.goto(x, y)  # Move turtle to new position
    turtle.setheading(i * 137.508)  # Set turtle heading for stamping or drawing
    turtle.pendown()

    if i < 160:
        turtle.stamp()  # Stamp the shape for first 160 iterations
    else:
        turtle.fillcolor("yellow")  # Change color for drawing
        turtle.begin_fill()
        turtle.left(-5)
        turtle.circle(500, 25)  # Draw part of the first circle
        turtle.right(-155)
        turtle.circle(500, 25)  # Draw part of the second circle
        turtle.end_fill()

# Hide the turtle when finished
turtle.hideturtle()
turtle.done()
