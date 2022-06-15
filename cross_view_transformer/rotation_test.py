from scipy.spatial.transform import Rotation as R
import numpy as np

# # CAM_FRONT_LEFT
# # cam from egocam
# # [0.82075834, -0.5712716, -0.002119458, -0.9649678]
# # [-0.00034143668, 0.0032195018, -0.99999475, 1.5082482]
# # [0.5712754, 0.82075477, 0.00244738, -1.280219]
# cam_from_egocam = np.array([
#         [0.82075834, -0.5712716, -0.002119458],
#         [-0.00034143668, 0.0032195018, -0.99999475],
#         [0.5712754, 0.82075477, 0.00244738]
# ])
# # E
# # [0.8133685, -0.581504, 0.016879816, -0.81018066]
# # [0.02294663, 0.003076115, -0.99973196, 1.5125322]
# # [0.58129615, 0.81353766, 0.01584557, -1.17453]
# E = np.array([
#         [0.8133685, -0.581504, 0.016879816],
#         [0.02294663, 0.003076115, -0.99973196],
#         [0.58129615, 0.81353766, 0.01584557]
# ])

# cam_from_egocam = R.from_matrix(cam_from_egocam)
# print(cam_from_egocam.as_euler('xyz',degrees=True))
# E = R.from_matrix(E)
# print(E.as_euler('xyz',degrees=True))











# r_front_left = R.from_quat([0.4998015430569128,
#         -0.5030316162024876,
#         0.4997798114386805,
#         -0.49737083824542755])
# matrix = r_front_left.as_matrix().T
# print("CAM_FRONT_LEFT \n" , matrix)
# euler = r_front_left.as_euler('xyz',degrees=True)
# print(euler)

# # CAM_FRONT
# r_front = R.from_quat([0.4998015430569128,
#         -0.5030316162024876,
#         0.4997798114386805,
#         -0.49737083824542755])
# matrix = r_front.as_matrix()
# print("CAM_FRONT\n", matrix)
# euler = r_front.as_euler('xyz',degrees=True)
# print(euler)

# # CAM_FRONT_RIGHT
# r_front_right = R.from_quat([0.2060347966337182,
#         -0.2026940577919598,
#         0.6824507824531167,
#         -0.6713610884174485])
# matrix = r_front_right.as_matrix()
# print("CAM_FRONT_RIGHT\n", matrix)
# euler = r_front_right.as_euler('xyz',degrees=True)
# print(euler)

# # CAM_BACK_LEFT
# r_back_left = R.from_quat([0.6924185592174665,
#         -0.7031619420114925,
#         -0.11648342771943819,
#         0.11203317912370753])
# matrix = r_back_left.as_matrix()
# print("CAM_BACK_LEFT\n", matrix)
# euler = r_back_left.as_euler('xyz',degrees=True)
# print(euler)

# # CAM_BACK
# r_back = R.from_quat([0.5037872666382278,
#         -0.49740249788611096,
#         -0.4941850223835201,
#         0.5045496097725578])
# matrix = r_back.as_matrix()
# print("CAM_BACK\n", matrix)
# euler = r_back.as_euler('xyz',degrees=True)
# print(euler)

# # CAM_BACK_RIGHT
# r_back_right = R.from_quat([0.12280980120078765,
#         -0.132400842670559,
#         -0.7004305821388234,
#         0.690496031265798])
# matrix = r_back_right.as_matrix()
# print("CAM_BACK_RIGHT\n", matrix)
# euler = r_back_right.as_euler('xyz',degrees=True)
# print(euler)







front = np.array([
        [-5.64133364e-03,-5.68014846e-03,  9.99967955e-01],
        [-9.99983763e-01,  8.37115272e-04, -5.63666773e-03],
        [-8.05071338e-04, -9.99983517e-01, -5.68477868e-03]
        ])
front_left = np.array([
        [ 2.44737995e-03,-8.20754770e-01,  5.71275430e-01],
        [-9.99994759e-01, -3.21950185e-03, -3.41436671e-04],
        [ 2.11945808e-03, -5.71271601e-01, -8.20758348e-01]
        ])


matrix =  front @ np.linalg.inv(front_left)
cam_from_egocam = R.from_matrix(matrix)
print(cam_from_egocam.as_euler('xyz',degrees=True))

cam_from_egocam = R.from_matrix(front)
print(cam_from_egocam.as_euler('xyz',degrees=True))

cam_from_egocam = R.from_matrix(front_left)
print(cam_from_egocam.as_euler('xyz',degrees=True))
# print(matrix)

# RESULT 8de7ec06e1ac48c689c4d24d6cc64fd7
# CAM_FRONT_LEFT 
#  [[ 2.44737995e-03 -8.20754770e-01  5.71275430e-01]
#  [-9.99994759e-01 -3.21950185e-03 -3.41436671e-04]
#  [ 2.11945808e-03 -5.71271601e-01 -8.20758348e-01]]

# [-1.45160964e+02 -1.21436094e-01 -8.98597750e+01]

# CAM_FRONT
#  [[-5.64133364e-03 -5.68014846e-03  9.99967955e-01]
#  [-9.99983763e-01  8.37115272e-04 -5.63666773e-03]
#  [-8.05071338e-04 -9.99983517e-01 -5.68477868e-03]]

# [-9.03257157e+01  4.61271948e-02 -9.03232264e+01]

# CAM_FRONT_RIGHT
#  [[-1.36479031e-02  8.32817742e-01  5.53379023e-01]
#  [-9.99865858e-01 -1.63788158e-02 -9.94603768e-06]
#  [ 9.05540984e-03 -5.53304927e-01  8.32929563e-01]]

# [-33.5956021   -0.51884386 -90.78202359]

# CAM_BACK_LEFT
#  [[-0.01601021 -0.94766474 -0.31886551]
#  [-0.99986478  0.0139763   0.00866572]
#  [-0.00375564  0.31896113 -0.94776036]]
# [161.39975386   0.21518276 -90.9173632 ]
# CAM_BACK
#  [[ 0.01674384 -0.00248837 -0.99985672]
#  [-0.99985181  0.00395911 -0.01675361]
#  [ 0.00400023  0.99998907 -0.00242171]]
# [ 90.138755    -0.22919686 -89.04059627]
# CAM_BACK_RIGHT
#  [[-0.01626597  0.93476883 -0.35488399]
#  [-0.99980932 -0.0113705   0.01587584]
#  [ 0.01080503  0.35507456  0.93477554]]
# [ 20.79928449  -0.61909476 -90.93206678]


# ipm___开始进行IPM处理 ipm_ba94cb79ebc74614bc2442185cb53c26.
# ipm___rotate_degrees [ 89.38857267 -35.8666618    1.43895241]
# ipm___rotate_degrees [-32.07101098 -88.99295686 122.51083997]
# ipm___rotate_degrees [-89.7942073  -32.92434595 178.60176067]
# ipm___rotate_degrees [91.67261499 18.51286675  1.69964722]
# ipm___rotate_degrees [112.35661916  89.53446037  21.78687717]
# ipm___rotate_degrees [-91.08015906  21.27491993 177.78345882]



# # cam from egocam
# ipm___rotate_degrees [ 8.98291520e+01 -3.48392129e+01 -2.38345765e-02]
# ipm___rotate_degrees [134.80330567 -89.54131031 -44.75626026]
# ipm___rotate_degrees [ -90.93885855  -33.59913759 -179.99931532]
# ipm___rotate_degrees [90.96788513 18.59433004  0.52386228]
# ipm___rotate_degrees [  8.45309652  89.03006562 -81.77493735]
# ipm___rotate_degrees [-90.99690535  20.78633473 179.02700642]

# # E
# ipm___rotate_degrees [ 88.8841641  -35.54176037   1.61599498]
# ipm___rotate_degrees [-15.25228169 -88.95973072 105.29295717]
# ipm___rotate_degrees [-90.08313004 -33.13474727 178.41349392]
# ipm___rotate_degrees [91.38033044 18.53362144  1.89785415]
# ipm___rotate_degrees [135.63723592  89.51285766  45.41741998]
# ipm___rotate_degrees [-91.50426513  21.11520202 177.61107771]