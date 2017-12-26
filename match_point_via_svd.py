import numpy as np
import matplotlib.pyplot as plt

def match_point_set(points_template, points_test):
    # http://nghiaho.com/?page_id=671
    points_template_np = points_template.copy()
    centroid_template_np = np.mean(points_template_np, axis = 0)
    points_template_np -= centroid_template_np

    points_test_np = points_test.copy()
    centroid_test_np = np.mean(points_test_np, axis = 0)
    points_test_np -= centroid_test_np
    npts = points_template_np.shape[0]
    points_test_t_np = points_test_np.transpose()
    cov_mat = np.zeros((2,2))
    for ii in range(0,npts):
        a= np.matrix(points_template_np[ii,:]).transpose()
        b=np.matrix(points_test_t_np[:,ii])
        c= a*b
        cov_mat += c
    u,s,v = np.linalg.svd(cov_mat)
    return v.T * u.T

points_template_np = np.array([[0,1.0],[2.0,2.0],[1.0,0]])
rot_angle = np.pi/4
#c,-s
#s,c
rot_mat_np = np.array([[np.cos(rot_angle),-np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]])
rotated_pts_np = rot_mat_np.dot(points_template_np.transpose()).transpose()

evaled_rot_mat_np = match_point_set(points_template_np,rotated_pts_np)
print("real rot mat:{}, \n evaled rot mat:{}".format(rot_mat_np,evaled_rot_mat_np))
guess_rotated_pt_np = evaled_rot_mat_np.dot((rotated_pts_np- np.mean(rotated_pts_np, axis = 0)).transpose()).transpose()
print(guess_rotated_pt_np)
fig = plt.figure()
plt.axis("equal")
plt.plot(points_template_np[:,0],points_template_np[:,1],"-go")
plt.plot(rotated_pts_np[:,0],rotated_pts_np[:,1],"-bo")
plt.plot(guess_rotated_pt_np[:,0],guess_rotated_pt_np[:,1],"-ro")
plt.show()