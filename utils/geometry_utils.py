# from ..main.src import params
import meep as mp
import numpy as np

# inicialize singleton of all parameters
# p = params.SimParams()

def make_cell(x=None, y=None, z=None, config=None):
    if config is not None:
        cell = mp.Vector3(config.cell_size[0],
            config.cell_size[1],
            config.cell_size[2])
    else:
        cell = mp.Vector3(x, y, z)
    return cell

# rounded edges !!!

def unit(v):
    return v / np.linalg.norm(v)

def flare_angle(P_center, P_left, P_right):
    v1 = P_left  - P_center
    v2 = P_right - P_center

    v1 = unit(v1)
    v2 = unit(v2)

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_theta)

def angle_bisector(P_center, P_left, P_right):
    v1 = unit(P_left  - P_center)
    v2 = unit(P_right - P_center)

    return unit(v1 + v2)

def inscribed_circle_center(P_center, P_left, P_right, R):
    theta = flare_angle(P_center, P_left, P_right)
    bis   = angle_bisector(P_center, P_left, P_right)

    L = R / np.sin(theta / 2.0)

    C = P_center + L * bis
    return C

def tangent_points(P_center, P_left, P_right, R):
    theta = flare_angle(P_center, P_left, P_right)
    d = R / np.tan(theta / 2.0)

    v1 = unit(P_left  - P_center)
    v2 = unit(P_right - P_center)

    T1 = P_center + d * v1
    T2 = P_center + d * v2
    return T1, T2

def safe_cut_point(P_center, P_left, P_right, R, overshoot=0.01):
    theta = flare_angle(P_center, P_left, P_right)
    bis   = angle_bisector(P_center, P_left, P_right)

    L = R / np.sin(theta / 2.0)

    # move slightly beyond the tangent point to avoid artifacts
    d = L - R * (1.0 - overshoot)

    return P_center - d * bis

def triangle_centroid(A, B, C):
    """
    Compute centroid (x, y) of triangle defined by points A, B, C.

    Parameters
    ----------
    A, B, C : np.ndarray shape (2,)
        Triangle vertices

    Returns
    -------
    np.ndarray shape (2,)
        Centroid coordinates
    """
    return (A + B + C) / 3.0

def clear_edges_bowtie(points, antenna, overshoot=0.01):
    """
    Clear all corners of a 2D polygon using air-cut prisms.

    Parameters
    ----------
    points : list of np.ndarray
        Polygon vertices in cyclic order, shape (N, 2)
    radius : float
        Fillet radius
    height : float
        Extrusion height
    overshoot : float
        Fraction of radius used to extend the air cut beyond tangency

    Returns
    -------
    geometry : list
        List of mp.GeometricObject (Prisms)
    """

    geometry = []
    N = len(points)

    r = antenna.radius
    h = antenna.thickness
    z = antenna.z_offset

    for i in range(N):
        P_center = points[i]
        P_left   = points[(i - 1) % N]
        P_right  = points[(i + 1) % N]

        # --- fillet geometry ---
        T1, T2 = tangent_points(P_center, P_left, P_right, r)
        P_cut  = safe_cut_point(P_center, P_left, P_right, r, overshoot)

        # --- air cut (corner removal) ---
        air_triangle = [
            mp.Vector3(T1[0], T1[1], 0),
            mp.Vector3(T2[0], T2[1], 0),
            mp.Vector3(P_cut[0], P_cut[1], 0)
        ]

        centroid = triangle_centroid(T1, T2, P_cut)

        geometry.append(
            mp.Prism(
                air_triangle,
                height=h,
                material=mp.air,
                center=mp.Vector3(centroid[0], centroid[1], z)
            )
        )

    return geometry

def fillet_bowtie(points, antenna, axis=np.array([0, 0, 1])):
    """
    Fillets all corners of a 2D polygon using material cylinders.

    Parameters
    ----------
    points : list of np.ndarray
        Polygon vertices in cyclic order, shape (N, 2)
    radius : float
        Fillet radius
    height : float
        Extrusion height
    material : mp.Medium
        Material of the solid polygon
    axis : array-like
        Extrusion axis (default z)

    Returns
    -------
    geometry : list
        List of mp.GeometricObject (Cylinders)
    """

    r = antenna.radius
    h = antenna.thickness
    z = antenna.z_offset

    geometry = []
    N = len(points)

    for i in range(N):
        P_center = points[i]
        P_left   = points[(i - 1) % N]
        P_right  = points[(i + 1) % N]

        # --- fillet geometry ---
        C = inscribed_circle_center(P_center, P_left, P_right, r)

        # --- fillet cylinder ---
        geometry.append(
            mp.Cylinder(
                center=mp.Vector3(C[0], C[1], z),
                radius=r,
                height=h,
                axis=mp.Vector3(*axis),
                material=antenna.material
            )
        )

    return geometry

def clear_rectangle_corners(points, antenna):
    """
    Remove triangular corners from rectangle.
    
    points : [(x,y)]
        rectangle vertices
    r : float
        cut size
    """

    geometry = []

    # środek prostokąta
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    r = antenna.radius
    h = antenna.thickness
    z = antenna.z_offset

    for (x, y) in points:

        sx = -1 if x > cx else 1
        sy = -1 if y > cy else 1

        P0 = (x, y)
        P1 = (x + sx * r, y)
        P2 = (x, y + sy * r)

        triangle = [
            mp.Vector3(P0[0], P0[1], 0),
            mp.Vector3(P1[0], P1[1], 0),
            mp.Vector3(P2[0], P2[1], 0)
        ]

        tx = (P0[0] + P1[0] + P2[0]) / 3
        ty = (P0[1] + P1[1] + P2[1]) / 3

        geometry.append(
            mp.Prism(
                triangle,
                height=h,
                material=mp.air,
                center=mp.Vector3(tx, ty, z)
            )
        )

    return geometry

def fillet_rectangle(points, antenna, axis=np.array([0,0,1])):

    geometry = []
    points = np.array(points)

    cx, cy = np.mean(points, axis=0)

    r = antenna.radius
    w = antenna.width
    if hasattr(antenna, "gap"):
        g = antenna.gap
    else:
        g = 0.0
    L = antenna.length
    x0, y0 = antenna.center
    z = antenna.z_offset

    def add_cyl(x, y):
        geometry.append(
            mp.Cylinder(
                center=mp.Vector3(x, y, z),
                radius=r,
                height=antenna.thickness,
                axis=axis,
                material=antenna.material
            )
        )

    # --- semicircle caps ---
    if r >= w/2:

        if g > 0:
            xs = [
                x0 + g/2 + r,
                x0 + g/2 + L - r,
                x0 - g/2 - r,
                x0 - g/2 - L + r,
            ]
        else:
            xs = [
                x0 + L/2 - r,
                x0 - L/2 + r,
            ]

        for x in xs:
            add_cyl(x, y0)

    # --- normal corner fillets ---
    else:

        for P in points:

            sx = -1 if P[0] > cx else 1
            sy = -1 if P[1] > cy else 1

            add_cyl(P[0] + sx*r, P[1] + sy*r)

    return geometry

def corrected_gap(g_target, R, theta):
        """
        Compute the nominal gap that must be used in geometry so that,
        after corner rounding with radius R, the effective gap equals g_target.

        Parameters
        ----------
        g_target : float
            Desired physical gap after rounding
        R : float
            Fillet radius
        theta : float
            Opening angle of the corner (radians)

        Returns
        -------
        g_input : float
            Gap to use in the sharp geometry
        """
        delta = R / np.sin(theta / 2.0) - R
        # print(f"Gap correction: For target gap {g_target*1e3:.2f} nm, radius {R*1e3:.1f} nm, and angle {np.rad2deg(theta):.1f} deg, the correction is {delta*1e3:.2f} nm.")
        return g_target - 2.0 * delta

def tip_apex_patch(antenna, x_length=None, y_height=None, overlap=2/1000):
    """
    Fill the thin air sliver that opens along the symmetry axis at the
    ROUNDED triangle apex of a HybridBar / BowTie tip.

    This used to be done in experiments.py with mp.Block "hole_fix" patches
    whose x-position (24 nm), size (26x6 nm) and z-size were HARD-CODED for
    a single case (gap 30 nm, W 240 nm, tip 150 nm). Here everything is
    derived from the antenna object, so the patch follows the tip for ANY
    gap / width / tip angle and for either metal layer (Au or Ti):

        x-centre = antenna.corected_gap        (24 nm for the old case;
                   corected_gap already folds in the radius correction)
        x-length = corected_gap + overlap      (~26 nm)   [override x_length]
        y-height = 1.2 * radius, min 4 nm       (~6 nm)    [override y_height]
        z-size   = antenna.thickness           (30 nm Au / 5 nm Ti)
        z-centre = antenna.z_offset            (layer position)
        material = antenna.material

    Call it ONLY when antenna.radius > 0 (no rounding -> no hole), and AFTER
    antenna.corected_gap has been computed inside build_geometry().

    NOTE: this is a numerical patch, not exact geometry. After a large change
    of gap/tip, verify visually with save_2D_plot that the hole is filled and
    the patch does not eat into the gap.

    Returns [right_patch, left_patch] (two mp.Block).
    """
    xc = antenna.corected_gap
    x0, y0 = antenna.center
    if x_length is None:
        x_length = antenna.corected_gap + overlap
    if y_height is None:
        y_height = max(1.2 * antenna.radius, 4/1000)

    return [
        mp.Block(
            mp.Vector3(x_length, y_height, antenna.thickness),
            center=mp.Vector3(x0 + xc, y0, antenna.z_offset),
            material=antenna.material,
        ),
        mp.Block(
            mp.Vector3(x_length, y_height, antenna.thickness),
            center=mp.Vector3(x0 - xc, y0, antenna.z_offset),
            material=antenna.material,
        ),
    ]