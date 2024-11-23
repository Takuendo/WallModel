#! /work/takumie/virtual/bin/python

import numpy as np
import shutil
import os
import glob
from scipy.integrate import odeint
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Execution.BasicRunner import BasicRunner
import logging
import sys
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import brentq

# ログの設定
logging.basicConfig(filename='script.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
from scipy.optimize import minimize_scalar

def find_tau_w(ix, iz, U_ODE, y_loc, iy_input_index,nu):
    def U_wall_function(tau_w):
        # 必要な定数や変数の定義
        kappa = 0.41
        A_plus = 17
        #nu = 0.009
        u_tau = np.sqrt(tau_w)
        U_LES = np.sqrt(U_ODE[ix, iy_input_index, iz, 0]**2 + U_ODE[ix, iy_input_index, iz, 2]**2)
        wall_shear_u = tau_w * U_ODE[ix, iy_input_index, iz, 0] / U_LES
        wall_shear_w = tau_w * U_ODE[ix, iy_input_index, iz, 2] / U_LES

        # U_ODE のコピーを作成
        U_temp = np.copy(U_ODE[ix,:,iz,:])

        # ボトム側の積分計算
        for k in range(iy_input_index, 0, -1):
            y_plus = y_loc[k] * u_tau / nu 
            nu_t_wm = kappa * u_tau * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
            delta_y = y_loc[k] - y_loc[k-1]
            U_temp[k-1,0] = U_temp[k,0] - delta_y / (nu + nu_t_wm) * wall_shear_u
            U_temp[k-1,2] = U_temp[k,2] - delta_y / (nu + nu_t_wm) * wall_shear_w

        # U_wall の計算
        y_plus = y_loc[0] * u_tau / nu 
        nu_t_wm = kappa * u_tau * y_loc[0] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        U_wall_value = U_temp[0,0] - y_loc[0] / (nu + nu_t_wm) * wall_shear_u
        return abs(U_wall_value)  # 目的関数として絶対値を返す

    # minimize_scalar を使用して tau_w を最適化
    result = minimize_scalar(U_wall_function, bounds=(1e-6, 4.0), method='bounded')
    if result.success:
        return result.x
    else:
        return None

def find_tau_w_top(ix, iz, U_ODE, y_loc, iy_input_index_top, ny,nu):
    def U_wall_function_top(tau_w_top):
        # トップ側の計算を行う
        kappa = 0.41
        A_plus = 17
        #nu = 0.009
        u_tau_top = np.sqrt(tau_w_top)
        U_LES_top = np.sqrt(U_ODE[ix, iy_input_index_top, iz, 0]**2 + U_ODE[ix, iy_input_index_top, iz, 2]**2)
        wall_shear_u_top = tau_w_top * U_ODE[ix, iy_input_index_top, iz, 0] / U_LES_top
        wall_shear_w_top = tau_w_top * U_ODE[ix, iy_input_index_top, iz, 2] / U_LES_top

        U_temp = np.copy(U_ODE[ix,:,iz,:])
        
        # トップ側の積分計算
        for l in range(iy_input_index_top+1, ny):
            k = ny-l
            y_plus = y_loc[k] * u_tau_top / nu 
            nu_t_wm = kappa * u_tau_top * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
            delta_y = y_loc[k]- y_loc[k-1]
            U_temp[l,0] = U_temp[l-1,0] - delta_y / (nu + nu_t_wm) * wall_shear_u_top
            U_temp[l,2] = U_temp[l-1,2] - delta_y / (nu + nu_t_wm) * wall_shear_w_top
        
        # U_wall の計算
        y_plus = y_loc[0] * u_tau_top / nu 
        nu_t_wm = kappa * u_tau_top * y_loc[0] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        U_wall_value_top = U_temp[ny-1,0] - y_loc[0] / (nu + nu_t_wm) * wall_shear_u_top
        return abs(U_wall_value_top)  

    # minimize_scalar を使用して tau_w を最適化
    result = minimize_scalar(U_wall_function_top, bounds=(1e-6, 4.0), method='bounded')
    if result.success:
        return result.x
    else:
        return None

def calculate_U_ODE(U_ODE, y_loc, ny, iy_input_index, iy_input_index_top, ix, iz, U_wall,nu):
    # tau_w と tau_w_top を計算
    tau_w = find_tau_w(ix, iz, U_ODE, y_loc, iy_input_index,nu)
    tau_w_top = find_tau_w_top(ix, iz, U_ODE, y_loc, iy_input_index_top, ny,nu)
    if ix == 0:
        if iz == 0:
            print('tau_w',tau_w)
            print('tau_w_top',tau_w_top)
    
    # tau_w または tau_w_top が見つからなかった場合の処理
    if tau_w is None:
        print(f"Could not find tau_w for ix={ix}, iz={iz}")
        return U_ODE, U_wall

    if tau_w_top is None:
        print(f"Could not find tau_w_top for ix={ix}, iz={iz}")
        return U_ODE, U_wall 
    # 以下、計算した tau_w と tau_w_top を用いて U_ODE と U_wall を計算
    
    # ボトム側の計算
    # 必要な変数の再計算
    kappa = 0.41
    A_plus = 17
    #nu = 0.009
    u_tau_bottom = np.sqrt(tau_w)
    U_LES = np.sqrt(U_ODE[ix, iy_input_index, iz, 0]**2 + U_ODE[ix, iy_input_index, iz, 2]**2)
    wall_shear_u = tau_w * U_ODE[ix, iy_input_index, iz, 0] / U_LES
    wall_shear_w = tau_w * U_ODE[ix, iy_input_index, iz, 2] / U_LES

    # ボトム側の積分計算
    for k in range(iy_input_index, 0, -1):
        y_plus = y_loc[k] * u_tau_bottom / nu 
        nu_t_wm = kappa * u_tau_bottom * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        delta_y = y_loc[k]- y_loc[k-1]
        U_ODE[ix,k-1,iz,0] = U_ODE[ix,k,iz,0] - delta_y / (nu + nu_t_wm) * wall_shear_u
        U_ODE[ix,k-1,iz,2] = U_ODE[ix,k,iz,2] - delta_y / (nu + nu_t_wm) * wall_shear_w

    # U_wall の計算
    y_plus = y_loc[0] * u_tau_bottom / nu 
    nu_t_wm = kappa * u_tau_bottom * y_loc[0] * (1 - np.exp(-y_plus / A_plus)) ** 2 
    U_wall[ix,0,iz,0] = U_ODE[ix,0,iz,0] - y_loc[0] / (nu + nu_t_wm) * wall_shear_u
    U_wall[ix,0,iz,2] = U_ODE[ix,0,iz,2] - y_loc[0] / (nu + nu_t_wm) * wall_shear_w

    # トップ側の計算
    # 必要な変数の再計算
    u_tau_top = np.sqrt(tau_w_top)
    U_LES_top = np.sqrt(U_ODE[ix, iy_input_index_top, iz, 0]**2 + U_ODE[ix, iy_input_index_top, iz, 2]**2)
    wall_shear_u_top = tau_w_top * U_ODE[ix, iy_input_index_top, iz, 0] / U_LES_top
    wall_shear_w_top = tau_w_top * U_ODE[ix, iy_input_index_top, iz, 2] / U_LES_top

    for l in range(iy_input_index_top+1, ny):
        k = ny-l
        y_plus = y_loc[k] * u_tau_top / nu 
        nu_t_wm = kappa * u_tau_top * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        delta_y = y_loc[k] - y_loc[k-1]
        U_ODE[ix,l,iz,0] = U_ODE[ix,l-1,iz,0] - delta_y / (nu + nu_t_wm) * wall_shear_u_top
        U_ODE[ix,l,iz,2] = U_ODE[ix,l-1,iz,2] - delta_y / (nu + nu_t_wm) * wall_shear_w_top

    # U_wall の計算
    y_plus = y_loc[0] * u_tau_top / nu 
    nu_t_wm = kappa * u_tau_top * y_loc[0] * (1 - np.exp(-y_plus / A_plus)) ** 2 
    U_wall[ix,1,iz,0] = U_ODE[ix,ny-1,iz,0] - y_loc[0] / (nu + nu_t_wm) * wall_shear_u_top
    U_wall[ix,1,iz,2] = U_ODE[ix,ny-1,iz,2] - y_loc[0] / (nu + nu_t_wm) * wall_shear_w_top
    if ix == 0:
        if iz == 0:
            print('wall_shear_u',wall_shear_u)
            print('wall_shear_w',wall_shear_w)
            print('wall_shear_u_top',wall_shear_u_top)
            print('wall_shear_w_top',wall_shear_w_top)
            print('U_wall[ix,0,iz,0]',U_wall[ix,0,iz,0])
            print('U_wall[ix,0,iz,2]',U_wall[ix,0,iz,2])
            print('U_wall[ix,1,iz,0]',U_wall[ix,1,iz,0])
            print('U_wall[ix,1,iz,2]',U_wall[ix,1,iz,2])
    return U_ODE, U_wall, wall_shear_u, wall_shear_w, wall_shear_u_top, wall_shear_w_top

"""
def calculate_U_ODE(U_ODE,y_loc,ny,iy_input_index,iy_input_index_top,ix,iz, U_wall):
    kappa = 0.41  # カルマン定数
    A_plus = 17
    nu = 0.009
    U_LES = np.sqrt(U_ODE[ix, iy_input_index, iz, 0]**2 + U_ODE[ix, iy_input_index, iz, 2]**2)
    U_LES_top np.sqrt(U_ODE[ix, ny-iy_input_index-1, iz, 0]**2 + U_ODE[ix, ny-iy_input_index-1, iz, 2]**2)
    wall_shear_u = tau_w * U_ODE[ix, iy_input_index, iz, 0] / U_LES
    wall_shear_w = tau_w * U_ODE[ix, iy_input_index, iz, 2] / U_LES
    wall_shear_u_top = tau_w_top * U_ODE[ix, ny-iy_input_index-, iz, 0] / U_LES_top
    wall_shear_w_top = tau_w_top * U_ODE[ix, ny-iy_input_index-, iz, 2] / U_LES_top
    u_tau_bottom = np.sqrt(tau_w)
    u_tau_top = np.sqrt(tau_w_top)

    #bottom
    for k in range(iy_input_index,0,-1):
        y_plus = y_loc[k] * u_tau_bottom / nu 
        nu_t_wm = kappa * u_tau_bottom * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        U_ODE[ix,k-1,iz,0] = U_ODE[ix,k,iz,0] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_u
        U_ODE[ix,k-1,iz,2] = U_ODE[ix,k,iz,0] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_w

    y_plus = y_loc[0] * u_tau_bottom / nu 
    nu_t_wm = kappa * u_tau_bottom * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
    U_wall[ix,0,iz,0] = U_ODE[ix,0,iz,0] - (y_loc[0])/(nu + nu_t_wm) * wall_shear_u
    U_wall[ix,0,iz,2] = U_ODE[ix,0,iz,0] - (y_loc[0])/(nu + nu_t_wm) * wall_shear_w 
    #top
    for l in range(iy_input_index_top+1, ny):
        k = ny - l + 1
        y_plus = y_loc[k] * u_tau_top / nu 
        nu_t_wm = kappa * u_tau_top * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        U_ODE[ix,l-1,iz,0] = U_ODE[ix,l,iz,0] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_u_top
        U_ODE[ix,l-1,iz,2] = U_ODE[ix,l,iz,0] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_w_top
    y_plus = y_loc[0] * u_tau_bottom / nu 
    nu_t_wm = kappa * u_tau_bottom * y_loc[0] * (1 - np.exp(-y_plus / A_plus)) ** 2 
    U_wall[ix,1,iz,0] = U_ODE[ix,ny,iz,0] - (y_loc[0])/(nu + nu_t_wm) * wall_shear_u_top
    U_wall[ix,1,iz,2] = U_ODE[ix,ny,iz,0] - (y_loc[0])/(nu + nu_t_wm) * wall_shear_w_top

    return U_ODE, U_wall

def compute_U_hwm(u_tau, nu, hwm):
    kappa = 0.41  # カルマン定数
    A_plus = 17
    def integrand(y):
        y_plus = y * u_tau / nu
        nu_t_wm = kappa * u_tau * y * (1 - np.exp(-y_plus / A_plus)) ** 2
        nu_total = nu + nu_t_wm
        return u_tau ** 2 / nu_total
    U_hwm, _ = quad(integrand, 0, hwm)
    return U_hwm

def residual(u_tau, nu, hwm, U_LES):
    U_hwm = compute_U_hwm(u_tau, nu, hwm)
    return U_hwm - U_LES

def solve_ode(rho, nu, hwm, U_LES):
    #U_LES = np.sqrt(u_LES**2 + w_LES**2)
    u_tau_guess = 1.35  # 初期推定値
    u_tau_solution = fsolve(residual, u_tau_guess, args=(nu, hwm, U_LES))
    u_tau = u_tau_solution[0]
    tau_w = rho * u_tau ** 2
    return tau_w

def calculate_U_ODE(U_ODE,wall_shear_u,wall_shear_w,wall_shear_u_top,wall_shear_w_top,y_loc,ny,iy_input_index,iy_input_index_top,tau_w,tau_w_top,ix,iz):
    kappa = 0.41  # カルマン定数
    A_plus = 17
    nu = 0.009
    u_tau_bottom = np.sqrt(tau_w)
    u_tau_top = np.sqrt(tau_w_top)

    mid =  iy_input_index // 2
    if ix == 0:
            if iz == 0:
                print('mid = ', mid)
    #bottom
    for k in range(0,mid):
        y_plus = y_loc[k] * u_tau_bottom / nu 
        nu_t_wm = kappa * u_tau_bottom * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        if k == 0:
            U_ODE[ix,0,iz,0] = (y_loc[0])/(nu + nu_t_wm) * wall_shear_u
            U_ODE[ix,0,iz,2] = (y_loc[0])/(nu + nu_t_wm) * wall_shear_w 
        else:
            U_ODE[ix,k,iz,0] = U_ODE[ix,k-1,iz,0] + (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_u
            U_ODE[ix,k,iz,2] = U_ODE[ix,k-1,iz,2] + (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_w
        if ix == 0:
            if iz == 0:
                print(k)
    for k in range(iy_input_index,mid,-1):
        y_plus = y_loc[k] * u_tau_bottom / nu 
        nu_t_wm = kappa * u_tau_bottom * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2  
        U_ODE[ix,k-1,iz,0] = U_ODE[ix,k,iz,0] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_u
        U_ODE[ix,k-1,iz,2] = U_ODE[ix,k,iz,2] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_w
        if ix == 0:
            if iz == 0:
                print(k)

    dy_forward = y_loc[mid + 1] - y_loc[mid]  # k と k+1 の間の距離
    dy_backward = y_loc[mid] - y_loc[mid - 1]  # k と k-1 の間の距離
    # 勾配から中間点の値を計算
    grad_u = (U_ODE[ix, mid + 1, iz, 0] - U_ODE[ix, mid - 1, iz, 0]) / (dy_forward + dy_backward)
    grad_w = (U_ODE[ix, mid + 1, iz, 2] - U_ODE[ix, mid - 1, iz, 2]) / (dy_forward + dy_backward)
    U_ODE[ix, mid, iz, 0] = U_ODE[ix, mid - 1, iz, 0] + grad_u * dy_backward
    U_ODE[ix, mid, iz, 2] = U_ODE[ix, mid - 1, iz, 2] + grad_w * dy_backward
    if ix == 0:
        if iz == 0:
            print(mid)

    #top
    for l in range(ny-1, ny-mid-1,-1):
        k = ny - l - 1
        y_plus = y_loc[k] * u_tau_top / nu 
        nu_t_wm = kappa * u_tau_top * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        if l == ny-1:
            U_ODE[ix,l,iz,0] = (y_loc[k])/(nu + nu_t_wm) * wall_shear_u
            U_ODE[ix,l,iz,2] = (y_loc[k])/(nu + nu_t_wm) * wall_shear_w 
        else:
            U_ODE[ix,l,iz,0] = U_ODE[ix,l+1,iz,0] + (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_u
            U_ODE[ix,l,iz,2] = U_ODE[ix,l+1,iz,2] + (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_w
        if ix == 0:
            if iz == 0:
                print(k)
                print(l)
    for l in range(iy_input_index_top,ny-mid-1):
        k = ny - l - 1
        y_plus = y_loc[k] * u_tau_top / nu 
        nu_t_wm = kappa * u_tau_top * y_loc[k] * (1 - np.exp(-y_plus / A_plus)) ** 2 
        U_ODE[ix,l+1,iz,0] = U_ODE[ix,l,iz,0] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_u_top
        U_ODE[ix,l+1,iz,2] = U_ODE[ix,l,iz,2] - (y_loc[k]-y_loc[k-1])/(nu + nu_t_wm) * wall_shear_w_top
        if ix == 0:
            if iz == 0:
                print(k)
                print(l)            

    dy_forward = y_loc[mid + 1] - y_loc[mid]  # k と k+1 の間の距離
    dy_backward = y_loc[mid] - y_loc[mid - 1]  # k と k-1 の間の距離
    # 勾配から中間点の値を計算
    grad_u = (U_ODE[ix, (ny-mid-1) + 1, iz, 0] - U_ODE[ix, (ny-mid-1) - 1, iz, 0]) / (dy_forward + dy_backward)
    grad_w = (U_ODE[ix, (ny-mid-1) + 1, iz, 2] - U_ODE[ix, (ny-mid-1) - 1, iz, 2]) / (dy_forward + dy_backward)
    U_ODE[ix, (ny-mid-1), iz, 0] = U_ODE[ix, (ny-mid-1) - 1, iz, 0] + grad_u * dy_backward
    U_ODE[ix, (ny-mid-1), iz, 2] = U_ODE[ix, (ny-mid-1) - 1, iz, 2] + grad_w * dy_backward
    if ix == 0:
        if iz == 0:
            print(mid)
            print(ny-mid-1)

    return U_ODE
"""
def get_velocities(case, time_step):
    U_file_path = os.path.join(case, time_step, "U")
    logging.debug(f'U_file_path: {U_file_path}')
    
    try:
        U_file = ParsedParameterFile(U_file_path)
        velocities = [Vector(*vec) for vec in U_file["internalField"]]
    except Exception as e:
        raise
    
    return velocities

def get_locations(case, feed_loc):
    points_file_path = os.path.join(case, "constant", "polyMesh", "C")
    
    with open(points_file_path, 'r') as f:
        lines = f.readlines()
        
    points_data = lines[lines.index('(\n') + 1 : lines.index(')\n')]
    
    locations = []
    y_loc = []
    for line in points_data:
        x, y, z = map(float, line.strip('()\n').split())
        locations.append(Vector(x, y, z))
        y_loc.append(y)
    
    y_loc = list(set(y_loc))
    for iy in y_loc:
        if abs(iy - feed_loc) == abs(np.array(y_loc) - feed_loc).min():
            iy_feed = iy

    return locations, iy_feed

def find_iy_feed_index(case, feed_loc):
    locations, iy_feed = get_locations(case, feed_loc)
    
    # y_loc 配列を取得してソート
    y_loc = sorted(set([loc[1] for loc in locations]))
    
    # iy_feed のインデックスを取得
    iy_feed_index = y_loc.index(iy_feed)
    
    return y_loc, iy_feed_index

def write_forces_to_case(velocities, forces, case, time_step, U_1):
    body_force_file_path = os.path.join(case, time_step, "U")
    os.makedirs(os.path.dirname(body_force_file_path), exist_ok=True)
    
    with open(body_force_file_path, 'w') as f:
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       volVectorField;\n')
        f.write('    location    "{}";\n'.format(time_step))
        f.write('    object      U;\n')
        f.write('}\n')
        f.write('dimensions      [0 1 -1 0 0 0 0];\n')
        f.write('internalField   nonuniform List<vector>\n')
        f.write('{}\n'.format(len(velocities)))
        f.write('(\n')

        for velocity in velocities:
            f.write('({} {} {})\n'.format(velocity[0], velocity[1], velocity[2]))
        f.write(')\n')
        f.write(';\n')
        f.write('boundaryField\n')
        f.write('{\n')
        f.write('    bottom\n')
        f.write('    {\n')
        f.write('        type            fixedGradient;\n') 
        f.write('        gradient        nonuniform List<vector>\n') 
        f.write('        {}\n'.format(len(forces)))
        f.write('        (\n')

        for force in forces:
            f.write('           ({} {} {})\n'.format(force[0], force[1], force[2]))

        f.write('        );\n')
        f.write('    }\n')
        f.write('    top\n')
        f.write('    {\n')
        f.write('        type            fixedGradient;\n') 
        f.write('        gradient        nonuniform List<vector>\n') 
        f.write('        {}\n'.format(len(forces)))
        f.write('        (\n')

        for force in U_1:
            f.write('           ({} {} {})\n'.format(force[0], force[1], force[2]))

        f.write('        );\n')
        f.write('    }\n')
        f.write('    left\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    right\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    inlet\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    outlet\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('}\n')

def write_only_velocity_to_case(velocities, case_dir, current_time):
    body_force_file_path = os.path.join(case_dir, current_time, "U")
    os.makedirs(os.path.dirname(body_force_file_path), exist_ok=True)
    
    with open(body_force_file_path, 'w') as f:
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       volVectorField;\n')
        f.write('    location    "{}";\n'.format(current_time))
        f.write('    object      U;\n')
        f.write('}\n')
        f.write('dimensions      [0 1 -1 0 0 0 0];\n')
        f.write('internalField   nonuniform List<vector>\n')
        f.write('{}\n'.format(len(velocities)))
        f.write('(\n')

        for velocity in velocities:
            f.write('({} {} {})\n'.format(velocity[0], velocity[1], velocity[2]))
        f.write(')\n')
        f.write(';\n')
        f.write('boundaryField\n')
        f.write('{\n')
        f.write('    bottom\n')
        f.write('    {\n')
        f.write('        type            noSlip;\n')
        f.write('    }\n')
        f.write('    top\n')
        f.write('    {\n')
        f.write('        type            noSlip;\n')
        f.write('    }\n')
        f.write('    left\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    right\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    inlet\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('    outlet\n')
        f.write('    {\n')
        f.write('        type            cyclic;\n')
        f.write('    }\n')
        f.write('}\n')

def main(case_dir, num_steps, save_steps, feed_loc, input_loc, feed_steps, nu, limit_steps, alpha, beta,input_loc_top):
    case = SolutionDirectory(case_dir)
    locations, iy_feed = get_locations(case_dir, feed_loc)
    locations_input, iy_input = get_locations(case_dir, input_loc)
    locations_input_top, iy_input_top = get_locations(case_dir, input_loc_top)
    y_loc, iy_feed_index = find_iy_feed_index(case_dir, feed_loc)
    y_loc, iy_input_index = find_iy_feed_index(case_dir, input_loc)
    y_loc_top, iy_input_index_top = find_iy_feed_index(case_dir, input_loc_top)

    current_time = case.getLast()
    start_step = int(float(current_time) / 0.0008) 

    for _ in range(start_step, start_step + num_steps):
        logging.debug(f'Step {_} start')
        current_time = case.getLast()
        logging.debug(f'Current time: {current_time}')
        velocities = get_velocities(case_dir, current_time)
        logging.debug(f'Velocities obtained at step {_}')

        U = np.zeros((32, 16, 32, 3))
        i = 0
        for iz in range(32):
            for iy in range(0,8):
                for ix in range(32):
                    U[ix, iy, iz, 0] = velocities[i][0]
                    U[ix, iy, iz, 1] = velocities[i][1]
                    U[ix, iy, iz, 2] = velocities[i][2]
                    i += 1
        print(i)
        for iz in range(32):
            for iy in range(8,16):
                for ix in range(32):
                    U[ix, iy, iz, 0] = velocities[i][0]
                    U[ix, iy, iz, 1] = velocities[i][1]
                    U[ix, iy, iz, 2] = velocities[i][2]
                    i += 1
        
        logging.debug(f'U array populated at step {_}')               

        print(iy_input_index)
        print(iy_input)
        print(iy_feed_index)
        print(iy_feed)
        print(y_loc[iy_input_index])
        U_input = U[:, iy_input_index, :, :]
        if _ % feed_steps == 0:
            INPUT_DB = np.zeros((32, 32, 3))
            OUTPUT_DB = np.zeros((32, 32, 6))
            Shear_u = np.zeros((32, 32))
            Shear_w = np.zeros((32, 32))
        INPUT_DB[:, :, :] = U_input

        wall_shears_u = []
        wall_shears_w = []
        wall_shears_u_top = []
        wall_shears_w_top = []
        gradient_u = []
        gradient_w = []  
        hwm = iy_input
        #nu = 0.009  # 動粘性係数
        rho = 1.0
        U_ODE = U[:, :, :, :]
        U_wall= np.zeros((32,2,32,3))
        ny = 16

        for ix in range(32):
            for iz in range(32):
                """
                #bottom
                U_LES = np.sqrt(U[ix, iy_input_index, iz, 0]**2 + U[ix, iy_input_index, iz, 2]**2)
                tau_w = solve_ode(rho, nu, hwm, U_LES)
                wall_shear_u = tau_w * U[ix, iy_input_index, iz, 0] / U_LES
                wall_shear_w = tau_w * U[ix, iy_input_index, iz, 2] / U_LES
                wall_shears_u.append(wall_shear_u)
                wall_shears_w.append(wall_shear_w)
                
                #top
                U_LES_top = np.sqrt(U[ix, iy_input_index_top, iz, 0]**2 + U[ix, iy_input_index_top, iz, 2]**2)
                tau_w_top = solve_ode(rho, nu, hwm, U_LES_top)
                wall_shear_u_top = tau_w_top * U[ix, iy_input_index, iz, 0] / U_LES_top
                wall_shear_w_top = tau_w_top * U[ix, iy_input_index, iz, 2] / U_LES_top
                wall_shears_u_top.append(wall_shear_u_top)
                wall_shears_w_top.append(wall_shear_w_top)
                """
                U_ODE, U_wall, wall_shear_u, wall_shear_w, wall_shear_u_top, wall_shear_w_top = calculate_U_ODE(U_ODE, y_loc, ny, iy_input_index, iy_input_index_top, ix, iz, U_wall,nu)
                wall_shears_u.append(wall_shear_u)
                wall_shears_w.append(wall_shear_w)
                wall_shears_u_top.append(wall_shear_u_top)
                wall_shears_w_top.append(wall_shear_w_top)
        
        wall_shears_u = np.array(wall_shears_u).reshape((32, 32))
        wall_shears_w = np.array(wall_shears_w).reshape((32, 32))
        wall_shears_u_top = np.array(wall_shears_u_top).reshape((32, 32))
        wall_shears_w_top = np.array(wall_shears_w_top).reshape((32, 32))
        
        ave_0 = np.mean(wall_shears_u[:,:])  # Compute the mean of all points
        std_0 = np.std(wall_shears_u[:,:])  # Compute the standard deviation of all points
        ave_0_top = np.mean(wall_shears_u_top[:,:]) 
        # Create the ave and std arrays with the same shape as the grid (32x32)
        ave_0_grid = np.full((32, 32), ave_0)
        std_0_grid = np.full((32, 32), std_0)

        # For the second layer (generated_data[:,:,1])
        ave_1 = np.mean(wall_shears_w[:,:])  # Compute the mean of all points
        std_1 = np.std(wall_shears_w[:,:])   # Compute the standard deviation of all points
        ave_1_top = np.mean(wall_shears_w_top[:,:]) 
        # Create the ave and std arrays with the same shape as the grid (32x32)
        ave_1_grid = np.full((32, 32), ave_1)
        std_1_grid = np.full((32, 32), std_1)


        # Ensure the relationship is satisfied: generated_data = ave + std
        #wall_shears_u  = alpha * ave_0_grid + beta * std_0_grid
        #wall_shears_w = alpha * ave_1_grid + beta * std_1_grid
        OUTPUT_DB[:, :, 0] =  - wall_shears_u / nu 
        OUTPUT_DB[:, :, 1] =  - (U[:, 1, :, 1]-U[:, 0, :, 1]) / (0.0271056-0.00756331) 
        OUTPUT_DB[:, :, 2] =  - wall_shears_w / nu 
        OUTPUT_DB[:, :, 3] =  - wall_shears_u_top / nu 
        OUTPUT_DB[:, :, 4] =    (U[:, 14, :, 1]-U[:, 15, :, 1]) / (0.0271056-0.00756331) 
        OUTPUT_DB[:, :, 5] =  - wall_shears_w_top / nu 

        with open('wallshear.txt', 'a') as f:  # 'a' モードで開くと加筆される
            f.write(f"{_}, {ave_0}, {ave_1}, {ave_0_top}, {ave_1_top}\n")
        
        logging.debug(f'Output DB updated at step {_}')    

        U_0 = []
        U_1 = []
        i = 0
        for iz in range(32):
            for ix in range(32):
                U_0.append((OUTPUT_DB[ix, iz, 0], OUTPUT_DB[ix, iz, 1], OUTPUT_DB[ix, iz, 2]))
                U_1.append((OUTPUT_DB[ix, iz, 3], OUTPUT_DB[ix, iz, 4], OUTPUT_DB[ix, iz, 5]))
                i += 1
        logging.debug(f'U_1 array populated at step {_}')   
        velocities_OUTPUT = np.zeros((32 * 16 * 32, 3))
        i = 0
        for iz in range(32):
            for iy in range(0,8):
                for ix in range(32):
                    velocities_OUTPUT[i][0] = U_ODE[ix, iy, iz, 0] 
                    velocities_OUTPUT[i][1] = U_ODE[ix, iy, iz, 1]
                    velocities_OUTPUT[i][2] = U_ODE[ix, iy, iz, 2] 
                    i += 1
        print(i)
        for iz in range(32):
            for iy in range(8,16):
                for ix in range(32):
                    velocities_OUTPUT[i][0] = U_ODE[ix, iy, iz, 0] 
                    velocities_OUTPUT[i][1] = U_ODE[ix, iy, iz, 1]
                    velocities_OUTPUT[i][2] = U_ODE[ix, iy, iz, 2] 
                    i += 1 

        write_forces_to_case(velocities_OUTPUT, U_0, case_dir, current_time, U_1)
        logging.debug(f'Forces written to case at step {_}')
        runner = BasicRunner(argv=["pimpleFoam", "-case", case_dir], silent=False)
        runner.start()
        logging.debug(f'Runner started at step {_}')
    
        if _ % save_steps != 0:
            shutil.rmtree(f"{case_dir}/{current_time}")
        
        for file in glob.glob(f"{case_dir}/PyFoam.pimpleFoam.logfile.restart*"):
            os.remove(file)
        
        if runner.runOK():
            case = SolutionDirectory(case_dir)
        else:
            break

if __name__ == "__main__":
    case_dir = "LES_co/"
    num_steps = 2000
    save_steps = 1
    feed_steps = 1
    limit_steps = 1
    nu= 0.009
    first_hight_coarse = 0.0271056
    feed_loc = 1.0 / 150.0
    input_loc_top = 1.9
    input_loc = 0.1
    alpha = 1
    beta = 0.5
    main(case_dir, num_steps, save_steps, feed_loc, input_loc, feed_steps, nu, limit_steps, alpha, beta,input_loc_top)
