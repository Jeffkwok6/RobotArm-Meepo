import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, cos, sin, simplify, pprint, pi
import sympy as sp
mpl.use('TkAgg')

def transzdn(dn):
    matrix= Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dn],
        [0, 0, 0, 1]
    ])
    return matrix
def rotz(theta):
    matrix= Matrix([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return matrix
def transxrn(rn):
    matrix= Matrix([
        [1, 0, 0 , rn],
        [0, 1 ,0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return matrix
def rotx(alpha):
    matrix= Matrix([
        [1, 0, 0 , 0],
        [0, cos(alpha), -sin(alpha), 0],
        [0, sin(alpha), cos(alpha), 0],
        [0, 0, 0, 1]
    ])
    return matrix
def HomogenousTransformation_matrix(params):
    d, theta, a, alpha = (params[0], params[1], params[2], params[3])
    alpha = alpha * pi / 180
    matrix = Matrix([transzdn(d)@rotz(theta)@transxrn(a)@rotx(alpha)])
    #print("Using DH_trans_matrix")
    #print(matrix)
    return matrix
def CompositeDH(DH_params):
    transforms = []
    # Identity Matrix
    transforms.append(sp.eye(4))
    for el in DH_params:
        #Add all the Homogenous Transformation Matrixs one at a time into Transforms List
        transforms.append(HomogenousTransformation_matrix(el))
        #print(transforms)
    return transforms
def jacobian_expression(DH_params):
    transforms = CompositeDH(DH_params)
    # Set initial frame to be the same as world basis vectors.
    Matrix_Transform = transforms[0]
    # For each matrix found in the transform list
    for matrix in transforms[1:]:
        # We are multiplying our previous frame by the next.
        # This to get a combined DH_Transform such as 0T1 x 1T2 x 2T3: -> 0T3
        Matrix_Transform = Matrix_Transform * matrix
    # Is the end effector position found by the translational 3x1 vector of the combined DH Matrix. Coloumn 4 Rows 1-3
    endeffector_position = Matrix_Transform[0:3,3]
    # The Jacobian usually includes at - least 6 rows because of the nature the partial derivative but can have less coloumns.
    # Deficient versus redundant robotic arm.
    J = sp.zeros(6,len(qcount))
    for joint in range(len(qcount)):
        # This method follows the notation Zi x (On - Oi) -> Local Partial Derivative.
        # End effector velocity relative to joint velocity.
        trans_joint = transforms[0]
        # Getting the combined DH transforms
        for matrix in transforms[1:joint+1]:
            # We are getting Oi in this for loop. We add +1 because joint starts from 0.
            trans_joint = trans_joint*matrix
            #print(trans_joint)
        # Getting the Zi matrix coloumn
        z_axis = trans_joint[0:3,2]
        # The On Values of the position Vecotrs in the DHtransform matrix
        joint_position = trans_joint[0:3,3]
        # Note this only works for revolute joints so far. This is Jv = Zi x (On - Oi)
        Jv = z_axis.cross(endeffector_position - joint_position)
        #print("This is the joint linear velocity for joint"+str(joint))
        #print(Jv)
        Jw = z_axis
        #print("This is the joint angular velocity for joint"+str(joint))
        #print(Jw)
        # Filling in the jacobian matrix. This method doesn't require defining the number of coloumns because we are simply filling in the rows by how many ever neede
        J[0:3, joint] = Jv
        J[3:6,joint] = Jw
    J = simplify(J)
    #print("This is the Jacobian for the input")
    #print(J)
    return J
def jacobian_angle_substitute(joints,jacobian_symbolic,count):
    # Conversion between input and ensure it's the correct dimensionality. (Sympy rules)
    if (isinstance(joints,np.ndarray)):
        joints = joints.flatten().tolist()
    # Getting the jacobian of interest and substituting the new values
    Jacobian_angle = jacobian_symbolic
    # For our purpose we delete joints 3 - 5
    for i,qi in enumerate(count,start = 0):
        Jacobian_angle = Jacobian_angle.subs(qi, joints[i])
        #print("Changing joint: "+str(i)+" to "+str(joints[i]))
        #print(Jacobian_angle)
    return Jacobian_angle
def CompositeDH_Eval(joints, DH_params,count):
    # Checking if right instance again
    if (isinstance(joints,np.ndarray)):
        joints = joints.flatten().tolist()
    # Getting DH transform
    transforms = CompositeDH(DH_params)
    # Identity matrix
    Matrix_Transform = transforms[0]
    for matrix in transforms[1:]:
        # Getting full DH transforms - Base to gripper
        Matrix_Transform = Matrix_Transform * matrix
    for i, qi in enumerate(count,start =0):
        Matrix_Transform = Matrix_Transform.subs(qi, joints[i])
    return Matrix_Transform
def joint_limits(joints):
    # These joint limits are in radians. Convert from degrees to radians. (Degrees*pi)/180
    # Joint 1
    if (joints[0] < -2 * pi):
        joints[0] = -2 * pi
    elif (joints[0] > 2 * pi):
        joints[0] = 2 * pi
    # Joint 2
    if (joints[1] < -220 * pi / 180):
        joints[1] = -220 * pi / 180
    elif (joints[1] > 50 * pi / 180):
        joints[1] = 50 * pi / 180
    # Joint 3
    if (joints[2] < - 50 * pi / 180):
        joints[2] = - 50 * pi / 180
    elif (joints[2] > 220 * pi / 180):
        joints[2] = 220 * pi / 180
    # Joint 4
    if (joints[3] < -2 * pi):
        joints[3] = -2 * pi
    elif (joints[3] > 2 * pi):
        joints[3] = 2 * pi
    # Joint 5
    if (joints[4] < - 50 * pi / 180):
        joints[4] = -50 * pi / 180
    elif (joints[4] > 220 * pi / 180):
        joints[4] = 220 * pi / 180
    return joints
def Inverse_Kinematics(joints, target, DH_params, error_trace=False,no_rotation=False,joint_lims=True):
    xr_desired = target[0:3, 0:3]
    xt_desired = target[0:3, 3]
    x_dot_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_trace = []
    iters = 0
    print("Finding symbolic jacobian")
    # Getting Symbolic Jacobian
    jacobian_symbolic = jacobian_expression(DH_params)
    final_xt = 0
    # While Running
    while(1):
        # Substituing joint values into jacombian symbolic
        jacobian = jacobian_angle_substitute(joints,jacobian_symbolic,qcount)
        jacobian = np.array(jacobian).astype(np.float64)
        # Subsituting joint values into full 0TN gripper DH Transform1
        Matrix_Transform = CompositeDH_Eval(joints, DH_params,qcount)
        # Conversion
        Matrix_Transform = np.array(Matrix_Transform).astype(np.float64)
        # Rotational Matrix
        xr_cur = Matrix_Transform[0:3,0:3]
        # Translation Matrix
        xt_cur = Matrix_Transform[0:3,3]
        # Final position of the end effector found from Xt_CUR
        final_xt = xt_cur
        # Target position minus the current position
        xt_dot = xt_desired - xt_cur
        # Desired position matrix multiplied by current translational position matrix
        R = xr_cur.T @ xr_desired
        # Angle of rotationa amount by finding Trace(R) - 1 / 2 SO3 -> $S$O3 log mapping
        v = np.arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
        r = (0.5 * 1/ np.sin(v)) * np.array([[R[2, 1] - R[1, 2]],
                                       [R[0, 2] - R[2, 0]],
                                       [R[1, 0] - R[0, 1]]])
        # Matrix Log?
        xr_dot = 200 * r * np.sin(v)
        xt_dot = xt_dot.reshape((3, 1))
        x_dot = np.vstack((xt_dot, xr_dot))
        x_dot_norm = np.linalg.norm(x_dot)
        if (x_dot_norm > 25):
            x_dot /= (x_dot_norm / 25)
        x_dot_change = np.linalg.norm(x_dot - x_dot_prev)
        if (x_dot_change < 0.0005):
            break
        x_dot_prev = x_dot
        e_trace.append(x_dot_norm)
        Lambda = 12
        # Damped Least Squares Method: Using Moore-Penrose Pseudo Inverse.
        # (JT*J + Lambda^2*I)^-1 * JT * dx
        joint_change = np.linalg.inv(jacobian.T@jacobian + Lambda**2*np.eye(len(qcount)))@jacobian.T@x_dot
        joints += joint_change
        if (joint_lims): joints  = joint_limits(joints)
        iters += 1
    print("Done in {} iterations".format(iters))
    print("Final Position is:")
    print(final_xt)
    for i, item in enumerate(joints, start  = 0):
        joints[i] = item * 180 / pi
    print(joints)
    return (joints, e_trace) if error_trace else joints
DH_params = []
spi = sp.pi
# Just add additional joints
q1, q2, q3, q4,q5 = symbols('q1 q2 q3 q4 q5')
# Must match the number of symbols you use
qcount = [q1, q2, q3,q4,q5]
# D & Theta & A & Alpha
DH_params.append([10, q1, 0, -90])
DH_params.append([0, q2, 10, 0])
DH_params.append([0, q3, 0, 90])
DH_params.append([10, q4, 0, 90])
DH_params.append([0, q5, 10, 90])
joints = np.array([[0.0], [0.0], [0.0],[0.0], [0.0]])
# Target positioning, make sure your xyz coloumns are the correct sign and XYZ does not exceed manipulability index
target = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 40],
                   [0, 0, 0, 1]])
new_j, e_trace = Inverse_Kinematics(joints, target, DH_params, error_trace=True)
