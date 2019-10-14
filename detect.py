import numpy as np
import cv2
import cv2.aruco as aruco
import math


def getCameraMatrix():
        with np.load('System.npz') as X:
                camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        return camera_matrix, dist_coeff


def sin(angle):
        return math.sin(math.radians(angle))


def cos(angle):
        return math.cos(math.radians(angle))



################################################################################


def detect_markers(img, camera_matrix, dist_coeff):
        markerLength = 100
        aruco_list = []
        ######################## INSERT CODE HERE ########################
        aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters=aruco.DetectorParameters_create()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners,ids,_=aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
        for j in range(len(ids)):
                for cor in corners[j]:
                        M=cv2.moments(cor)
                        cx=math.ceil(M['m10']/M['m00'])
                        cy=math.ceil(M['m01']/M['m00'])
                        l=[cx,cy]
                rvec,tvec,_=aruco.estimatePoseSingleMarkers(corners[j],100,camera_matrix,dist_coeff)
                aruco_list.append([ids[j],tuple(l),rvec,tvec])
                aruco_list[j]=tuple(aruco_list[j])
        ##################################################################
        return aruco_list


def drawAxis(img, aruco_list, aruco_id, camera_matrix, dist_coeff):
        for x in aruco_list:
                if aruco_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        m = markerLength/2
        pts = np.float32([[-m,m,0],[m,m,0],[-m,-m,0],[-m,m,m]])
        pt_dict = {}
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        for i in range(len(pts)):
                 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
        src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
        dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
        
        img = cv2.line(img, src, dst1, (0,255,0), 4)
        img = cv2.line(img, src, dst2, (255,0,0), 4)
        img = cv2.line(img, src, dst3, (0,0,255), 4)
        return img


def drawCube(img, ar_list, ar_id, camera_matrix, dist_coeff):
        for x in ar_list:
                if ar_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        m = markerLength/2
        ######################## INSERT CODE HERE ########################  
        pts = np.float32([[-m,m,0],[m,m,0],[m,-m,0],[-m,-m,0],[-m,m,2*m],[m,m,2*m],[m,-m,2*m],[-m,-m,2*m]])
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        imgpts=np.int32(imgpts).reshape(-1,2)
        img=cv2.drawContours(img,[imgpts[:4]],-1,(0,0,255),2)
        for i,j in zip(range(4),range(4,8)):
                img=cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(0,0,255),2)
        img=cv2.drawContours(img,[imgpts[4:]],-1,(0,0,255),2) 
        ##################################################################
        return img


def drawCylinder(img, ar_list, ar_id, camera_matrix, dist_coeff):
        for x in ar_list:
                if ar_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        radius = markerLength/2; height = markerLength*1.5
        ######################## INSERT CODE HERE ########################
        m=int(radius)
        h=height
        pts=np.float32([[0,-m,0],[-0.5*m,-0.85*m,0],[-0.85*m,-0.5*m,0],[-m,0,0],[-0.85*m,0.5*m,0],[-0.5*m,0.85*m,0],
                        [0,m,0],[0.5*m,0.85*m,0],[0.85*m,0.5*m,0],[m,0,0],[0.85*m,-0.5*m,0],[0.5*m,-0.85*m,0],
                        [0,-m,h],[-0.5*m,-0.85*m,h],[-0.85*m,-0.5*m,h],[-m,0,h],[-0.85*m,0.5*m,h],[-0.5*m,0.85*m,h],
                        [0,m,h],[0.5*m,0.85*m,h],[0.85*m,0.5*m,h],[m,0,h],[0.85*m,-0.5*m,h],[0.5*m,-0.85*m,h]])
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        imgpts=np.int32(imgpts).reshape(-1,2)
        for i,j in zip(range(6),range(6,12)):
            img=cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255,0,0),2)
        for i,j in zip(range(12,18),range(18,24)):
            img=cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255,0,0),2)
        for i,j in zip(range(12),range(12,24)):
            img=cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255,0,0),2)
        cord=[]
        cord1=[]
        for angle in range(0,360):
            theta=math.radians(angle)
            x=m*math.cos(theta)
            y=m*math.sin(theta)
            cord.append([x,y,0])
            cord1.append([x,y,h])
        imgpts, _ = cv2.projectPoints(np.float32([cord]), rvec, tvec, camera_matrix, dist_coeff)
        imgpts=np.int32(imgpts).reshape(-1,2)
        img=cv2.drawContours(img,[imgpts[:len(imgpts)]],-1,(255,0,0),2)
        imgpts, _ = cv2.projectPoints(np.float32([cord1]), rvec, tvec, camera_matrix, dist_coeff)
        imgpts=np.int32(imgpts).reshape(-1,2)
        img=cv2.drawContours(img,[imgpts[:len(imgpts)]],-1,(255,0,0),2)
        ##################################################################
        return img


if __name__=="__main__":
        cam, dist = getCameraMatrix()
        img = cv2.imread("..\\TestCases\\image_4.jpg")
        aruco_list = detect_markers(img, cam, dist)
        for i in aruco_list:
                img = drawAxis(img, aruco_list, i[0], cam, dist)
                img = drawCube(img, aruco_list, i[0], cam, dist)
                img = drawCylinder(img, aruco_list, i[0], cam, dist)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
