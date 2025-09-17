import cv2
import numpy as np
import matplotlib.pyplot as plt    

# create a blank image
img = np.zeros((960, 960, 3), dtype=np.uint8)

for k in range(0, 300):
  # give a random colour to the image
  # img[:] = np.random.randint(0, 255, 3)
  img[:] = (255, 255, 255)

  # define minimum and maximum size of the shape
  minRatio, maxRatio = 0.05, 0.2
  lb = int(min(img.shape[:2]) * minRatio)
  ub = int(min(img.shape[:2]) * maxRatio)

  shapes = np.array([3, 4, 20])     # triangle, square, circle

  # lists to save the coordinates already occupied by shapes
  xymin = []
  xymax = []

  # select random number of shapes (between 2 to 7)
  for i in range(np.random.randint(2, 7)):
    
    radius = np.random.randint(lb, ub)

    # number of sides
    idx = np.random.randint(0, len(shapes))
    n = shapes[idx]
    # angle
    angle = np.deg2rad(np.random.randint(0, 360))
    # colour
    colour = tuple(np.random.randint(0, 255, 3).tolist())

    center = np.random.randint(radius, min(img.shape[:2]) - radius, 2).astype(float)

    # generate n points radius distance away from the centre at equal angles
    points = []
    for j in range(n):
        theta = angle + j * (2*np.pi/n)  # 360/n degrees apart
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    xmin, ymin = points.min(axis = 0)
    xmax, ymax = points.max(axis = 0)

    # cv2.rectangle(img, (xmin-15, ymin-15), (xmax+15, ymax+15), (255, 255, 255), -1)

    # draw filled shape
    cv2.drawContours(img, [points], 0, colour, -1)

    # saving image
    cv2.imwrite(f"D:\git\cv-project-log\object\datasets\shapes\images\w_{k}.jpg", img)

    # get coordinates for the bounding box 
    # xmin, ymin = points.min(axis = 0)
    # xmax, ymax = points.max(axis = 0)



    w = (xmax - xmin) / img.shape[0]      # yolo uses normalised coordinates for its anotations
    h = (ymax - ymin) / img.shape[1]
    xcenter = center[0] / img.shape[0]
    ycenter = center[1] / img.shape[1]

    # creating/appending annotation file with same name as the image
    with open(f"D:\git\cv-project-log\object\datasets\shapes\labels\w_{k}.txt", "a") as f:
      f.write(f"{idx} {xcenter} {ycenter} {w}  {h}\n")
    
#   plt.imshow(img)
#   plt.axis("off")
#   plt.show()
