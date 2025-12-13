"""
Code By John Meshinsky and Beom Jun Kim

Uses modifed code from MAE 263F Lectures
"""

import numpy as np
import matplotlib.pyplot as plt

def crossMat(a):
    """Returns the cross product matrix of vector 'a'."""
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return A

def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """Returns the derivative of bending energy."""
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])
    kappaBar = curvature0
    gradKappa = np.zeros(6)
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    te = ee / norm_e
    tf = ef / norm_f
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))
    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi
    kappa1 = kb[2]
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k
    return dF

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """Returns the Hessian of bending energy."""
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])
    kappaBar = curvature0
    gradKappa = np.zeros(6)
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    te = ee / norm_e
    tf = ef / norm_f
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))
    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi
    kappa1 = kb[2]
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]
    DDkappa1 = np.zeros((6, 6))
    norm2_e = norm_e**2
    norm2_f = norm_f**2
    Id3 = np.eye(3)
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e = np.outer(kb, m2e)
    D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
                  kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
                  (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)
    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)
    D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - crossMat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * EI * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * EI * DDkappa1
    return dJ

def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """Calculate the gradient of the stretching energy."""
    F = np.zeros(4)
    F[0] = -(1.0 - np.sqrt((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0) / l_k) * \
           ((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0)**(-0.5) / l_k * (-2.0 * xkp1 + 2.0 * xk)
    F[1] = -(1.0 - np.sqrt((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k) * \
           ((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5) / l_k * (-2.0 * ykp1 + 2.0 * yk)
    F[2] = -(1.0 - np.sqrt((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k) * \
           ((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5) / l_k * (2.0 * xkp1 - 2.0 * xk)
    F[3] = -(1.0 - np.sqrt((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k) * \
           ((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5) / l_k * (2.0 * ykp1 - 2.0 * yk)
    F = 0.5 * EA * l_k * F
    return F

def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """Returns the 4x4 Hessian of the stretching energy."""
    J = np.zeros((4, 4))
    J11 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (-2 * xkp1 + 2 * xk)**2) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * ((-2 * xkp1 + 2 * xk)**2) / 2 - \
          2 * (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5)) / l_k
    J12 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * (-2 * xkp1 + 2 * xk) * (-2 * ykp1 + 2 * yk) / 2
    J13 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * xkp1 - 2 * xk) / 2 + \
          2 * (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5)) / l_k
    J14 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * ykp1 - 2 * yk) / 2
    J22 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (-2 * ykp1 + 2 * yk)**2) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * ((-2 * ykp1 + 2 * yk)**2) / 2 - \
          2 * (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5)) / l_k
    J23 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * xkp1 - 2 * xk) / 2
    J24 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * ykp1 - 2 * yk) / 2 + \
          2 * (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5)) / l_k
    J33 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * xkp1 - 2 * xk)**2) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * ((2 * xkp1 - 2 * xk)**2) / 2 - \
          2 * (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5)) / l_k
    J34 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * (2 * xkp1 - 2 * xk) * (2 * ykp1 - 2 * yk) / 2
    J44 = (1 / ((xkp1 - xk)**2 + (ykp1 - yk)**2) / l_k**2 * (2 * ykp1 - 2 * yk)**2) / 2 + \
          (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-1.5)) / l_k * ((2 * ykp1 - 2 * yk)**2) / 2 - \
          2 * (1 - np.sqrt(((xkp1 - xk)**2 + (ykp1 - yk)**2)) / l_k) * \
          (((xkp1 - xk)**2 + (ykp1 - yk)**2)**(-0.5)) / l_k
    J = np.array([[J11, J12, J13, J14],
                   [J12, J22, J23, J24],
                   [J13, J23, J33, J34],
                   [J14, J24, J34, J44]])
    J *= 0.5 * EA * l_k
    return J

def create_triangle_lattice(L_edge, N_per_edge, nx_triangles, ny_triangles, apex_height_ratio=0.75):
    """
    Creates a lattice of connected four-edged triangles (with internal apex).
    Each triangle has 4 beams: 2 outer edges + 2 internal edges to apex.

    Parameters:
    -----------
    L_edge : float
        Base edge length of each triangle
    N_per_edge : int
        Number of nodes per edge (used for spacing calculation)
    nx_triangles : int
        Number of triangles in x direction
    ny_triangles : int
        Number of triangles in y direction
    apex_height_ratio : float
        Height ratio for apex position

    Returns:
    --------
    nodes : ndarray
        All node positions
    beam_list : list
        List of beams (each beam is a list of node indices)
    boundary_info : dict
        Information about boundary nodes
    """

    all_nodes = []
    node_map = {}
    beam_list = []
    outer_nodes = {}  # Track outer star nodes: (row, col, vertex) -> node_idx
    inner_nodes = {}  # Track inner star nodes: (row, col, edge) -> node_idx

    tolerance = 1e-10

    # Outer star square size (reference square from center)
    outer_size = L_edge * apex_height_ratio
    # Inner star square size (half of outer)
    inner_size = outer_size / 2.0
    # Origin square extends from -L_edge/2 to +L_edge/2

    def get_or_create_node(pos):
        """Get existing node or create new one."""
        for existing_pos, idx in node_map.items():
            if np.linalg.norm(np.array(pos) - np.array(existing_pos)) < tolerance:
                return idx
        idx = len(all_nodes)
        all_nodes.append(pos)
        node_map[tuple(pos)] = idx
        return idx

    def create_beam(start, end, N):
        """Create beam with N evenly spaced nodes."""
        beam_nodes = []
        for i in range(N):
            t = i / (N - 1)
            node = start + t * (end - start)
            idx = get_or_create_node(node)
            beam_nodes.append(idx)
        return beam_nodes

    # Create grid of stars
    for row in range(ny_triangles):
        for col in range(nx_triangles):
            # Center of this star's origin square
            center_x = col * L_edge
            center_y = row * L_edge
            center_pos = np.array([center_x, center_y])

            # Outer star square vertices (4 nodes)
            outer_half = outer_size / 2.0
            outer_vertices = {
                'top_right': center_pos + np.array([outer_half, outer_half]),
                'bottom_right': center_pos + np.array([outer_half, -outer_half]),
                'bottom_left': center_pos + np.array([-outer_half, -outer_half]),
                'top_left': center_pos + np.array([-outer_half, outer_half])
            }

            # Inner star square edge midpoints (4 nodes)
            inner_half = inner_size / 4.0
            inner_edge_midpoints = {
                'top': center_pos + np.array([0, inner_half]),
                'right': center_pos + np.array([inner_half, 0]),
                'bottom': center_pos + np.array([0, -inner_half]),
                'left': center_pos + np.array([-inner_half, 0])
            }

            # Create nodes for outer vertices
            for vertex_name, vertex_pos in outer_vertices.items():
                vertex_idx = get_or_create_node(vertex_pos)
                outer_nodes[(row, col, vertex_name)] = vertex_idx

            # Create nodes for inner edge midpoints
            for edge_name, edge_pos in inner_edge_midpoints.items():
                edge_idx = get_or_create_node(edge_pos)
                inner_nodes[(row, col, edge_name)] = edge_idx

            # Connect inner nodes to adjacent outer nodes (NOT to other inner nodes)
            # Each inner node connects to the 2 outer vertices it's between
            connections = {
                'top': ['top_left', 'top_right'],      # top inner connects to top_left and top_right outer
                'right': ['top_right', 'bottom_right'], # right inner connects to top_right and bottom_right outer
                'bottom': ['bottom_right', 'bottom_left'], # bottom inner connects to bottom_right and bottom_left outer
                'left': ['bottom_left', 'top_left']     # left inner connects to bottom_left and top_left outer
            }

            for inner_edge, outer_verts in connections.items():
                inner_idx = inner_nodes.get((row, col, inner_edge))
                for outer_vert in outer_verts:
                    outer_idx = outer_nodes.get((row, col, outer_vert))
                    if inner_idx is not None and outer_idx is not None:
                        beam = create_beam(all_nodes[inner_idx], all_nodes[outer_idx], N_per_edge)
                        beam_list.append(beam)

            # Extend beam from each inner node outward (away from center) to edge of origin square
            # Origin square extends from center ± L_edge/2
            origin_edge = L_edge / 2.0
            outward_directions = {
                'top': np.array([0, 1]),      # Up
                'right': np.array([1, 0]),    # Right
                'bottom': np.array([0, -1]),  # Down
                'left': np.array([-1, 0])     # Left
            }

            for inner_edge, direction in outward_directions.items():
                inner_idx = inner_nodes.get((row, col, inner_edge))
                if inner_idx is not None:
                    inner_pos = all_nodes[inner_idx]
                    # Calculate point on origin square edge in this direction
                    # Find intersection of ray from center through inner node with origin square boundary
                    # Origin square: center ± origin_edge in both x and y
                    center_to_inner = inner_pos - center_pos
                    # Normalize direction
                    if np.linalg.norm(center_to_inner) > tolerance:
                        # The edge point is at center + direction * origin_edge
                        edge_point = center_pos + direction * origin_edge
                        beam = create_beam(inner_pos, edge_point, N_per_edge)
                        beam_list.append(beam)

    nodes = np.array(all_nodes)

    # Find boundary nodes
    y_coords = nodes[:, 1]
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    bottom_nodes = []
    top_nodes = []

    for i in range(len(nodes)):
        if abs(nodes[i, 1] - min_y) < tolerance:
            bottom_nodes.append(i)
        if abs(nodes[i, 1] - max_y) < tolerance:
            top_nodes.append(i)

    boundary_info = {
        'bottom_nodes': bottom_nodes,
        'top_nodes': top_nodes
    }

    return nodes, beam_list, boundary_info

def objfun(qOld, uOld, freeIndex, ndof, dt, tol, massVector, mMat,
          EA, EI, beam_list, deltaL,
          F_applied):

  q = qOld.copy()
  iter_count = 0
  error = 10 * tol
  max_iter = 100

  while error > tol and iter_count < max_iter:
      Fb = np.zeros(ndof)
      Jb = np.zeros((ndof, ndof))

      # Bending forces for each beam
      for beam in beam_list:
          for k in range(1, len(beam) - 1):
              node_km1 = beam[k - 1]
              node_k = beam[k]
              node_kp1 = beam[k + 1]

              xkm1 = q[2*node_km1]
              ykm1 = q[2*node_km1 + 1]
              xk = q[2*node_k]
              yk = q[2*node_k + 1]
              xkp1 = q[2*node_kp1]
              ykp1 = q[2*node_kp1 + 1]

              ind = [2*node_km1, 2*node_km1+1, 2*node_k, 2*node_k+1, 2*node_kp1, 2*node_kp1+1]

              gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)
              hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)
              Fb[ind] -= gradEnergy
              Jb[np.ix_(ind, ind)] -= hessEnergy

      # Stretching forces for each beam
      Fs = np.zeros(ndof)
      Js = np.zeros((ndof, ndof))
      for beam in beam_list:
          for k in range(len(beam) - 1):
              node_k = beam[k]
              node_kp1 = beam[k + 1]

              xk = q[2*node_k]
              yk = q[2*node_k + 1]
              xkp1 = q[2*node_kp1]
              ykp1 = q[2*node_kp1 + 1]

              ind = [2*node_k, 2*node_k+1, 2*node_kp1, 2*node_kp1+1]

              gradEnergy = gradEs(xk, yk, xkp1, ykp1, deltaL, EA)
              hessEnergy = hessEs(xk, yk, xkp1, ykp1, deltaL, EA)
              Fs[ind] -= gradEnergy
              Js[np.ix_(ind, ind)] -= hessEnergy

      Forces = Fb + Fs + F_applied
      JForces = Jb + Js

      f = massVector / dt * ((q - qOld) / dt - uOld) - Forces
      J = mMat / dt**2 - JForces

      f_free = f[freeIndex]
      J_free = J[np.ix_(freeIndex, freeIndex)]

      dq_free = np.linalg.solve(J_free, f_free)
      q[freeIndex] = q[freeIndex] - dq_free

      # Enforce boundary conditions:
      # - All bottom nodes fixed in y
      # - All top nodes fixed in x
      for node_idx in bottom_nodes:
          q[2 * node_idx + 1] = q0[2 * node_idx + 1]
      # for node_idx in top_nodes:
          # q[2 * node_idx] = q0[2 * node_idx]

      error = np.sum(np.abs(f_free))
      iter_count += 1

  u = (q - q0) / dt
  return q, u

# Material properties - TPU (Thermoplastic Polyurethane)
L = 0.3  # Smaller triangles for lattice
R = 0.015  # Outer radius
r = 0  # Inner radius
E = 26e6  # Young's modulus for TPU (26 MPa = 26e6 Pa), flexible material
rho = 1200  # Density of TPU (kg/m^3)
A = np.pi * (R**2 - r**2)
I = np.pi * (R**4 - r**4) / 4
EA = E * A
EI = E * I

print("=== MATERIAL PROPERTIES (TPU) ===")
print(f"Young's Modulus: {E/1e6:.1f} MPa")
print(f"Density: {rho} kg/m^3")
print(f"Cross-sectional area: {A*1e6:.3f} mm^2")
print(f"Second moment of area: {I*1e12:.3f} mm^4")
print(f"EA: {EA:.3e} N")
print(f"EI: {EI:.3e} N·m^2")

# Create lattice (3x3 pattern)
N_per_edge = 6
nx_tri = 5
ny_tri = 5
nodes_init, beam_list, boundary_info = create_triangle_lattice(
    L, N_per_edge, nx_tri, ny_tri
)

N_total = len(nodes_init)
ndof = 2 * N_total

print("\n=== LATTICE STRUCTURE ===")
print(f"Lattice size: {nx_tri}x{ny_tri} triangles")
print(f"Total nodes: {N_total}")
print(f"Total DOF: {ndof}")
print(f"Number of beams: {len(beam_list)}")
print(f"Bottom fixed nodes: {len(boundary_info['bottom_nodes'])}")
print(f"Top force nodes: {len(boundary_info['top_nodes'])}")

# Calculate spacing
edge_lengths_per_beam = []
for beam in beam_list:
    if len(beam) > 1:
        beam_length = 0
        for i in range(len(beam) - 1):
            node1 = nodes_init[beam[i]]
            node2 = nodes_init[beam[i + 1]]
            seg_length = np.linalg.norm(node2 - node1)
            beam_length += seg_length
        edge_lengths_per_beam.append(beam_length / (len(beam) - 1))

deltaL = np.mean(edge_lengths_per_beam)
print(f"Average segment length (deltaL): {deltaL:.6f} m")

total_length = 0
for beam in beam_list:
    for i in range(len(beam) - 1):
        node1 = nodes_init[beam[i]]
        node2 = nodes_init[beam[i + 1]]
        total_length += np.linalg.norm(node2 - node1)
print(f"Total beam length: {total_length:.3f} m")

# Identify the two middle bottom nodes for force application
# (do this before plotting so force_nodes is defined)
bottom_nodes = boundary_info['bottom_nodes']
top_nodes = boundary_info['top_nodes']

# Sort bottom nodes by x-coordinate to find the middle ones
bottom_nodes_sorted = sorted(bottom_nodes, key=lambda idx: nodes_init[idx, 0])
n_bottom = len(bottom_nodes_sorted)

force_nodes = top_nodes

print(f"\nForce will be applied to nodes: {force_nodes}")
print(f"Positions: {[nodes_init[i] for i in force_nodes]}")

# Plot initial lattice
fig, ax = plt.subplots(figsize=(12, 12))
for beam in beam_list:
    x_beam = [nodes_init[i, 0] for i in beam]
    y_beam = [nodes_init[i, 1] for i in beam]
    ax.plot(x_beam, y_beam, 'b-', linewidth=1.5, alpha=0.6)

ax.plot(nodes_init[:, 0], nodes_init[:, 1], 'ko', markersize=3)

# Highlight force nodes with force
ax.plot(nodes_init[force_nodes, 0], nodes_init[force_nodes, 1], 'g^',
        markersize=12, label='Force (+y)', markeredgewidth=2, zorder=5)

ax.plot(nodes_init[bottom_nodes, 0], nodes_init[bottom_nodes, 1], 'rs',
        markersize=10, label='Fixed y-only', markeredgewidth=2)

# Draw arrows showing force direction on the two middle bottom nodes
arrow_length = 0.03
for node_idx in force_nodes:
    ax.arrow(nodes_init[node_idx, 0], nodes_init[node_idx, 1],
             0, arrow_length, head_width=0.015, head_length=0.015,
             fc='green', ec='green', linewidth=2, alpha=0.7, zorder=6)

ax.set_xlabel('x (m)', fontsize=14)
ax.set_ylabel('y (m)', fontsize=14)
ax.set_title('Initial Lattice Structure (TPU Material)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('star_lattice_initial_structure.png', dpi=300, bbox_inches='tight')
plt.show()

# Setup simulation
m = rho * A * total_length / N_total
massVector = np.full(ndof, m)
mMat = np.diag(massVector)

# Initial DOF
q0 = np.zeros(ndof)
q0[0::2] = nodes_init[:, 0]
q0[1::2] = nodes_init[:, 1]
u = np.zeros(ndof)

# Fixed DOF:
# - All bottom nodes: fixed only in y-direction
# - All top nodes: fixed in x-direction (but free in y where force is applied)
fixedIndex = []
for node_idx in bottom_nodes:  # All bottom nodes fixed in y
    fixedIndex.append(2 * node_idx + 1)  # Only fix y-coordinate
for node_idx in top_nodes:  # All top nodes fixed in x
    fixedIndex.append(2 * node_idx)  # Only fix x-coordinate
fixedIndex = np.array(fixedIndex)
freeIndex = np.setdiff1d(np.arange(ndof), fixedIndex)

# Applied force: 10 N in +y at the top nodes
F_applied = np.zeros(ndof)

total_applied_force = 200.0
force_per_node = total_applied_force / len(top_nodes)  # Apply to top nodes
force_per_node_val = force_per_node
for node_idx in top_nodes:  # top_nodes get the force
    F_applied[2 * node_idx + 1] = force_per_node

print(f"\n=== SIMULATION SETUP ===")
print(f"Bottom nodes (fixed in y only): {len(bottom_nodes)} nodes")
print(f"Top nodes (fixed in x, force in y): {len(top_nodes)} nodes")
print(f"Force per top node: {force_per_node:.1f} N (+y)")
print(f"Total force: {force_per_node * len(top_nodes):.1f} N")
print(f"Free DOF: {len(freeIndex)} out of {ndof}")

# Time parameters
totalTime = 0.35 # Changed to 1 second
dt = 0.0005
Nsteps = int(totalTime / dt)
tol = EI / L**2 * 1e-3

# tracking arrays
time_history = []
force_history = []
height_history = []
width_history = []
poisson_history = []

displacement_top_history = []
displacement_top_avg_history = []
work_done_history = []
shock_absorption_coeff_history = []

# Get initial configuration
x_init =q0[0::2]
y_init = q0[1::2]
initial_height = np.max(y_init) - np.min(y_init)
initial_width = np.max(x_init) - np.min(x_init)

initial_top_y_coords = [y_init[idx] for idx in top_nodes]
initial_avg_top_y = np.mean(initial_top_y_coords)



# Snapshots
num_snapshots = 10
snapshot_times = np.linspace(0, totalTime, num_snapshots)
snapshot_indices = (snapshot_times / dt).astype(int)
snapshots = []


#Simulation Loop
print(f"\nStarting simulation: {totalTime}s total, dt={dt}s, {Nsteps} steps")

ctime = 0
for step in range(Nsteps):
    if step % 100 == 0:
        print(f"Time: {ctime:.2f}s / {totalTime}s")

    q = q0.copy()

    qNew, uNew = objfun(q, u, freeIndex, ndof, dt, tol, massVector, mMat,
           EA, EI, beam_list, deltaL,
           F_applied)

    q0 = qNew.copy()
    u = uNew.copy()

    ## SHOCK ABSORPTION ANALYSIS OVER TIME - Calculations
    # Calculate current structure height (deformation metric)
    x_nodes_snap = q0[0::2] # Renamed to avoid conflict with x_nodes
    y_nodes_snap = q0[1::2] # Renamed to avoid conflict with y_nodes

    width = np.max(x_nodes_snap) - np.min(x_nodes_snap)
    height = np.max(y_nodes_snap) - np.min(y_nodes_snap)
    width_history.append(width)
    height_history.append(height)

    # Calculate displacement of top nodes (where force is applied)
    current_top_y_positions = []
    for node_idx in top_nodes:
        current_top_y_positions.append(y_nodes_snap[node_idx])
    current_avg_top_y = np.mean(current_top_y_positions)

    # Displacement is positive in +y direction (extension/compression)
    displacement_top = current_avg_top_y - initial_avg_top_y
    displacement_top_history.append(displacement_top)
    displacement_top_avg_history.append(displacement_top)

    # Calculate work done by applied force
    # Work is positive when force and displacement are in same direction
    work_done = total_applied_force * displacement_top
    work_done_history.append(work_done)

    # Calculate instantaneous Poisson's ratio
    long_strain = (height - initial_height) / initial_height
    trans_strain = (width - initial_width) / initial_width
    nu = 0.0
    if abs(long_strain) > 1e-10:
        nu = -trans_strain / long_strain
    poisson_history.append(nu)

    # time tracker
    time_history.append(ctime)

    if step in snapshot_indices:
        snapshots.append((ctime, q.copy()))

    ctime += dt

print("Simulation complete!")


# SHOCK ABSORPTION ANALYSIS OVER TIME - Calculations
#################

# Calculate maximum values for shock absorption
max_displacement = np.max(displacement_top_history)
max_work = np.max(work_done_history)

# Calculate shock absorption efficiency
# Efficiency = (Energy absorbed through deformation) / (Input energy)
# Input energy = Force × Maximum displacement
input_energy = 0
absorbed_energy = 0
if abs(max_displacement) > 1e-10:
    input_energy = total_applied_force * abs(max_displacement)
    absorbed_energy = max_work


# FINISHING POISSON'S RATIO CALCULATIONS
#################

# Get final configuration
x_final = q0[0::2]
y_final = q0[1::2]

# Calculate initial dimensions
x_coords_init = x_init
y_coords_init = y_init
initial_width = np.max(x_coords_init) - np.min(x_coords_init)
# initial_height already calculated

# Calculate final dimensions
x_coords_final = x_final
y_coords_final = y_final
final_width = np.max(x_coords_final) - np.min(x_coords_final)
final_height = np.max(y_coords_final) - np.min(y_coords_final)

# Calculate height deformation
height_deformation = np.array(height_history) - initial_height
height_deformation_percent = (height_deformation / initial_height) * 100.0
max_deformation = np.max(height_deformation)

# Calculate strains
longitudinal_strain = (final_height - initial_height) / initial_height  # Strain in loading direction (y)
transverse_strain = (final_width - initial_width) / initial_width  # Strain perpendicular to loading (x)

# Poisson's ratio: nu = -transverse_strain / longitudinal_strain
poisson_ratio = 0.0
if abs(longitudinal_strain) > 1e-10:
    poisson_ratio = -transverse_strain / longitudinal_strain


# ALL PLOTS
#################

# Plot snapshots
fig, axes = plt.subplots(2, 5, figsize=(20, 16))
axes = axes.flatten()

for idx, (t, q_snap) in enumerate(snapshots):
    ax = axes[idx]
    x_nodes = q_snap[0::2]
    y_nodes = q_snap[1::2]

    # Plot beams
    for beam in beam_list:
        x_beam = [x_nodes[i] for i in beam]
        y_beam = [y_nodes[i] for i in beam]
        ax.plot(x_beam, y_beam, 'b-', linewidth=1.5, alpha=0.6)

    # Plot nodes
    ax.plot(x_nodes, y_nodes, 'ko', markersize=3)

    # Highlight boundary nodes
    # Highlight fixed y nodes (bottom nodes)
    ax.plot(x_nodes[bottom_nodes], y_nodes[bottom_nodes], 'rs',
            markersize=8, markeredgewidth=2)
    # Highlight force nodes (which are top nodes)
    ax.plot(x_nodes[force_nodes], y_nodes[force_nodes], 'g^',
            markersize=10, markeredgewidth=2)

    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_title(f't = {t:.2f} s', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    if idx == 0:
        ax.legend(['Beam', 'Nodes', 'Fixed y-only', 'Force (+y)'],
                  fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('star_lattice_deformation_snapshots.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot initial vs final
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Initial configuration
ax = axes[0]
x_init_plot = snapshots[0][1][0::2] # Renamed to avoid conflict
y_init_plot = snapshots[0][1][1::2] # Renamed to avoid conflict
for beam in beam_list:
    x_beam = [x_init_plot[i] for i in beam]
    y_beam = [y_init_plot[i] for i in beam]
    ax.plot(x_beam, y_beam, 'b-', linewidth=2, alpha=0.6)
ax.plot(x_init_plot, y_init_plot, 'ko', markersize=4)
# Make fixed nodes (bottom nodes) red squares
ax.plot(x_init_plot[bottom_nodes], y_init_plot[bottom_nodes], 'rs',
        markersize=10, label='Fixed y-only', markeredgewidth=2)
# Make force nodes (top nodes) green triangles
ax.plot(x_init_plot[force_nodes], y_init_plot[force_nodes], 'g^',
        markersize=12, label='Force (+y)', markeredgewidth=2)
ax.set_xlabel('x (m)', fontsize=14)
ax.set_ylabel('y (m)', fontsize=14)
ax.set_title('Initial Configuration', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend(fontsize=12)

# Final configuration
ax = axes[1]
x_final_plot = snapshots[-1][1][0::2] # Renamed to avoid conflict
y_final_plot = snapshots[-1][1][1::2] # Renamed to avoid conflict
for beam in beam_list:
    x_beam = [x_final_plot[i] for i in beam]
    y_beam = [y_final_plot[i] for i in beam]
    ax.plot(x_beam, y_beam, 'r-', linewidth=2, alpha=0.8)
ax.plot(x_final_plot, y_final_plot, 'ro', markersize=4)
# Make fixed nodes (bottom nodes) red squares
ax.plot(x_final_plot[bottom_nodes], y_final_plot[bottom_nodes], 'rs',
        markersize=10, label='Fixed y-only', markeredgewidth=2)
# Make force nodes (top nodes) green triangles
ax.plot(x_final_plot[force_nodes], y_final_plot[force_nodes], 'g^',
        markersize=12, label='Force (+y)', markeredgewidth=2)
ax.set_xlabel('x (m)', fontsize=14)
ax.set_ylabel('y (m)', fontsize=14)
ax.set_title(f'Final Configuration (t={totalTime}s)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('star_lattice_initial_vs_final.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot dimensional changes over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Width vs time
ax = axes[0, 0]
ax.plot(time_history, np.array(width_history), 'b-', linewidth=2)
ax.axhline(initial_width, color='k', linestyle='--', label='Initial width')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Width (m)', fontsize=12)
ax.set_title('Structure Width Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Height vs time
ax = axes[0, 1]
ax.plot(time_history, np.array(height_history), 'r-', linewidth=2)
ax.axhline(initial_height, color='k', linestyle='--', label='Initial height')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Height (m)', fontsize=12)
ax.set_title('Structure Height Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Strains vs time
ax = axes[1, 0]
long_strain_history = [(h - initial_height) / initial_height * 100 for h in height_history]
trans_strain_history = [(w - initial_width) / initial_width * 100 for w in width_history]
ax.plot(time_history, long_strain_history, 'r-', linewidth=2, label='Longitudinal (y)')
ax.plot(time_history, trans_strain_history, 'b-', linewidth=2, label='Transverse (x)')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Strain (%)', fontsize=12)
ax.set_title('Strain Evolution', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Poisson's ratio vs time
ax = axes[1, 1]
ax.plot(time_history, poisson_history, 'g-', linewidth=2)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel("Poisson's Ratio (ν)", fontsize=12)
ax.set_title("Poisson's Ratio Over Time", fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('star_poisson_ratio_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#################
# SHOCK ABSORPTION ANALYSIS PLOTS
#################

fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # Changed to 1 row, 3 columns

# Plot 1: Height Deformation vs Time
ax = axes[0]
ax.plot(time_history, np.array(height_deformation), 'b-', linewidth=2, label='Height Change')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Height Deformation (m)', fontsize=12)
ax.set_title('Structure Height Deformation Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 2: Top Node Displacement vs Time
ax = axes[1]
ax.plot(time_history, np.array(displacement_top_history), 'g-', linewidth=2, label='Top Node Displacement')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Displacement (m)', fontsize=12)
ax.set_title('Top Node Displacement (Force Application Point)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 3: Work Done vs Time
ax = axes[2]
ax.plot(time_history, np.array(work_done_history), 'orange', linewidth=2, label='Work Done')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Work Done (J)', fontsize=12)
ax.set_title('Work Done by Applied Force Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('star_shock_absorption_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#################
# ALL PRINTED ANALYSIS
#################

print("\n=== SHOCK ABSORPTION ANALYSIS ===")
print(f"Total applied force: {total_applied_force:.3f} N")
print(f"Force applied to {len(top_nodes)} top nodes")
print(f"\nInitial structure height: {initial_height:.6f} m")
print(f"Initial average top node y-position: {initial_avg_top_y:.6f} m")

print(f"\n=== SHOCK ABSORPTION RESULTS ===")
print(f"Maximum top node displacement: {max_displacement:.6f} m")
print(f"Maximum height deformation: {max_deformation:.6f} m ({max_deformation/initial_height*100:.2f}%)")
print(f"Maximum work done: {max_work:.3f} J")
print(f"Final displacement: {displacement_top_history[-1]:.6f} m")
print(f"Final height deformation: {height_deformation[-1]:.6f} m")

if abs(max_displacement) > 1e-10:
    print(f"  Input energy: {input_energy:.3f} J")
    print(f"  Absorbed energy: {absorbed_energy:.3f} J")

print("\n=== POISSON'S RATIO CALCULATION ===")
print(f"Initial dimensions:")
print(f"  Width (x): {initial_width:.6f} m")
print(f"  Height (y): {initial_height:.6f} m")
print(f"\nFinal dimensions:")
print(f"  Width (x): {final_width:.6f} m")
print(f"  Height (y): {final_height:.6f} m")
print(f"\nDimensional changes:")
print(f"  ΔWidth: {(final_width - initial_width):.6f} m")
print(f"  ΔHeight: {(final_height - initial_height):.6f} m")
print(f"\nStrains:")
print(f"  Longitudinal strain (ε_y): {longitudinal_strain:.6f} ({longitudinal_strain*100:.3f}%)")
print(f"  Transverse strain (ε_x): {transverse_strain:.6f} ({transverse_strain*100:.3f}%)")
print(f"\nPoisson's ratio (ν): {poisson_ratio:.4f}")

# All bottom nodes - should only move in x (y is fixed)
avg_x_disp_bottom = 0
avg_y_disp_bottom = 0
for node_idx in bottom_nodes:  # All bottom nodes are fixed in y
    dx = x_final[node_idx] - x_init[node_idx]
    dy = y_final[node_idx] - y_init[node_idx]
    avg_x_disp_bottom += abs(dx)
    avg_y_disp_bottom += abs(dy)
avg_x_disp_bottom /= len(bottom_nodes)
avg_y_disp_bottom /= len(bottom_nodes)

# All top nodes - should only move in y (x is fixed)
avg_x_disp_top = 0
avg_y_disp_top = 0
for node_idx in top_nodes:  # All top nodes are fixed in x
    dx = x_final[node_idx] - x_init[node_idx]
    dy = y_final[node_idx] - y_init[node_idx]
    avg_x_disp_top += abs(dx)
    avg_y_disp_top += abs(dy)
avg_x_disp_top /= len(top_nodes)
avg_y_disp_top /= len(top_nodes)

# Max displacement
max_disp = 0
max_disp_node = 0
for i in range(N_total):
    dx = x_final[i] - x_init[i]
    dy = y_final[i] - y_init[i]
    disp = np.sqrt(dx**2 + dy**2)
    if disp > max_disp:
        max_disp = disp
        max_disp_node = i

print("\n=== FINAL RESULTS ===")
avg_disp_top = 0
for node_idx in top_nodes:
    dx = x_final[node_idx] - x_init[node_idx]
    dy = y_final[node_idx] - y_init[node_idx]
    disp = np.sqrt(dx**2 + dy**2)
    avg_disp_top += disp
avg_disp_top /= len(top_nodes)
print(f"Average displacement of top nodes (with force): {avg_disp_top:.6f} m")
print(f"\nFixed bottom nodes (y fixed):")
print(f"  Average x displacement: {avg_x_disp_bottom:.6f} m")
print(f"  Average y displacement: {avg_y_disp_bottom:.9f} m (should be ~0)")
print(f"\nTop nodes (x fixed, force in y):")
print(f"  Average x displacement: {avg_x_disp_top:.9f} m (should be ~0)")
print(f"  Average y displacement: {avg_y_disp_top:.6f} m")
print(f"\nMaximum displacement: {max_disp:.6f} m at node {max_disp_node}")