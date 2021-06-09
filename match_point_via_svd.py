import math 
import numpy as np
import matplotlib.pyplot as plt
# http://nghiaho.com/?page_id=671

fig = plt.figure()
axes1 = plt.subplot(111)
axes1.axis("equal")

original_pts_np = np.mat([[0,1.5],[2.0,2.0],[1.5,0]])
rot_angle = np.pi/4.0
c = math.cos(rot_angle)
s = math.sin(rot_angle)
ground_truth_rot_mat = np.mat([[c,-s],[s,c]])
pts_after_rotate_np = (ground_truth_rot_mat * original_pts_np.T).T

axes1.plot(original_pts_np[:,0],original_pts_np[:,1],"-go")
axes1.plot(pts_after_rotate_np[:,0],pts_after_rotate_np[:,1],"-ro")
axes1.annotate("before rotate", xy=(original_pts_np[1,0], original_pts_np[1,1]))
axes1.annotate("after rotate", xy=(pts_after_rotate_np[1,0], pts_after_rotate_np[1,1]))

covariance_mat = original_pts_np.T * pts_after_rotate_np
u,s,v = np.linalg.svd(covariance_mat)
# u, v:(2, 2)
guess_rot_mat= v.T* u.T

vec1_a = np.stack([np.array([[0,0]]),u[0,:]])
vec1_b = np.stack([np.array([[0,0]]), u[1,:]])
axes1.plot(vec1_a[:,0],vec1_a[:,1],"-g")
axes1.plot(vec1_b[:,0],vec1_b[:,1],"-g")
axes1.annotate("u", xy=(u[0,0], u[0,1]))

vec2_a = np.stack([np.array([[0,0]]),v[0,:]])
vec2_b = np.stack([np.array([[0,0]]), v[1,:]])
axes1.plot(vec2_a[:,0],vec2_a[:,1],"-r")
axes1.plot(vec2_b[:,0],vec2_b[:,1],"-r")
axes1.annotate("v", xy=(v[1,0], v[1,1]))


print(ground_truth_rot_mat - guess_rot_mat)

plt.show()