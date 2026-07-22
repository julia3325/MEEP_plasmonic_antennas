from abc import ABC, abstractmethod
import numpy as np
import meep as mp
from meep.materials import Au, Ti, SiO2
from utils.geometry_utils import *
xm = 1000

# =========================================================
# Base class
# =========================================================

class AntennaBase(ABC):

    @abstractmethod
    def build_geometry(self):
        pass

    @abstractmethod
    def bounding_box(self):
        pass

# =========================================================
# HybridBar
# =========================================================

class HybridBar(AntennaBase):
    def __init__(self,
                 gap,
                 bar_length,
                 tip_length,
                 width,
                 thickness,
                 material,
                 z_offset=0.0,
                 center=(0.0, 0.0),
                 radius=0.0):

        self.gap = gap
        self.corected_gap = gap 
        self.bar_length = bar_length
        self.tip_length = tip_length
        self.width = width
        self.thickness = thickness
        self.material = material
        self.z_offset = z_offset
        self.center = np.array(center)
        self.radius = radius

    def build_geometry(self):
        
        # Wyliczenie skorygowanej szczeliny w przypadku zaokrąglenia
        if self.radius > 1e-12:
            angle = np.arctan(self.width / (2 * self.tip_length)) * 2
            self.corected_gap = corrected_gap(
                self.gap,
                self.radius,
                angle)
        else:
            self.corected_gap = self.gap

        overlap = 2/1000 # 2nm zakladki, aby uniknac artefaktow numerycznych pomiedzy dwoma czesciami anten

        # Punkty dla prawej strony trójkąta
        P1 = np.array([self.center[0] + self.corected_gap/2.0, 0])
        P2 = P1 + np.array([self.tip_length, self.width/2.0])
        P3 = P1 + np.array([self.tip_length, -self.width/2.0])

        tip_points_right = [mp.Vector3(*P1), mp.Vector3(*P2), mp.Vector3(*P3)]
        
        # Punkty dla lewej strony trójkąta
        mirror = np.array([-1.0, 1.0])
        tip_points_left = [mp.Vector3(*(P1 * mirror)), 
                           mp.Vector3(*(P2 * mirror)), 
                           mp.Vector3(*(P3 * mirror))]
        
        x_centroid = (P1[0] + P2[0] + P3[0]) / 3.0

        # Współrzędna X, w której kończy się trójkąt, by poprawnie podpiąć prostokąt (bar)
        tip_end_x = self.center[0] + self.corected_gap/2.0 + self.tip_length

        geometry = [
            mp.Prism(tip_points_right, height=self.thickness, material=self.material, 
                     center=mp.Vector3(x_centroid, 0, self.z_offset)),
            mp.Prism(tip_points_left, height=self.thickness, material=self.material,
                     center=mp.Vector3(-x_centroid, 0, self.z_offset)),
            mp.Block(
                mp.Vector3(self.bar_length + overlap, self.width, self.thickness),
                center=mp.Vector3(tip_end_x + self.bar_length/2.0 - overlap/2.0,
                                  self.center[1],
                                  self.z_offset),
                material=self.material
            ),
            mp.Block(
                mp.Vector3(self.bar_length + overlap, self.width, self.thickness),
                center=mp.Vector3(-tip_end_x - self.bar_length/2.0 + overlap/2.0,
                                  self.center[1],
                                  self.z_offset),
                material=self.material
            )
        ]

        # Dodanie zaokrągleń (fillets) na rogach
        if self.radius > 1e-12:
            geometry += clear_edges_bowtie(
                points=[P1, P2, P3],
                antenna=self,
            )
            geometry += clear_edges_bowtie(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                antenna=self,
            )
            geometry += fillet_bowtie(
                points=[P1, P2, P3],
                antenna=self
            )
            geometry += fillet_bowtie(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                antenna=self
            )

        return geometry

    def bounding_box(self):
        return [
            2 * (self.bar_length + self.tip_length) + self.corected_gap,
            self.width,
            self.thickness
        ]

# =========================================================
# BowTieEquilateral
# =========================================================

class BowTieEquilateral(AntennaBase):

    def __init__(self,
                 gap,
                 length,
                 thickness,
                 radius,
                 material,
                 z_offset=0.0,
                 center=(0.0, 0.0)):

        self.gap = gap
        self.corected_gap = gap
        self.length = length
        self.width = 2*length*np.tan(np.deg2rad(30))
        self.thickness = thickness
        self.radius = radius
        self.material = material
        self.z_offset = z_offset
        self.center = np.array(center)

    # -----------------------------------------------------

    def build_geometry(self):

        if self.radius > 1e-12:
            self.corected_gap = corrected_gap(
                self.gap,
                self.radius,
                np.deg2rad(60))
            # print(f"Corrected gap for radius {self.radius*1e3:.1f} nm: {self.gap*1e3:.2f} nm")

        # Right triangle tip
        P1 = np.array([
            self.center[0] + self.corected_gap/2.0,
            self.center[1]
        ])

        # Equilateral triangle assumption
        P2 = P1 + self.length * np.array([
            1.0,
            np.tan(np.deg2rad(30))
        ])

        P3 = P2 * np.array([1.0, -1.0])

        tip_right = [
            mp.Vector3(*P1),
            mp.Vector3(*P2),
            mp.Vector3(*P3)
        ]

        # Mirror on Y axis
        mirror = np.array([-1.0, 1.0])

        tip_left = [
            mp.Vector3(*(P1 * mirror)),
            mp.Vector3(*(P2 * mirror)),
            mp.Vector3(*(P3 * mirror))
        ]

        x_centroid = self.length * 2/3 + self.corected_gap/2.0
        bow_tie = [
            mp.Prism(
                tip_right,
                height=self.thickness,
                material=self.material,
                center=mp.Vector3(x_centroid, 0, self.z_offset)
            ),
            mp.Prism(
                tip_left,
                height=self.thickness,
                material=self.material,
                center=mp.Vector3(-x_centroid, 0, self.z_offset)
            )
        ]

        # Fillets / rounding
        if self.radius > 1e-12:

            bow_tie += clear_edges_bowtie(
                points=[P1, P2, P3],
                antenna=self
            )

            bow_tie += clear_edges_bowtie(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                antenna=self
            )

            bow_tie += fillet_bowtie(
                points=[P1, P2, P3],
                antenna=self
            )

            bow_tie += fillet_bowtie(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                antenna=self
            )

        return bow_tie

    # -----------------------------------------------------

    def bounding_box(self):

        return [
            2*self.length + self.gap,
            self.width,
            self.thickness
        ]

# =========================================================
# BowTie
# =========================================================

class BowTie(AntennaBase):

    def __init__(self,
                 gap,
                 length,
                 width,
                 thickness,
                 material,
                 z_offset=0.0,
                 center=(0.0, 0.0),
                 radius=0.0):

        self.gap = gap
        self.corected_gap = gap
        self.length = length
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.material = material
        self.z_offset = z_offset
        self.center = np.array(center)

    # -----------------------------------------------------

    def build_geometry(self):

        if self.radius > 1e-12:
            angle = np.arctan(self.width/(2*self.length)) * 2
            self.corected_gap = corrected_gap(
                self.gap,
                self.radius,
                angle)
            # self.corected_gap = self.gap
            # print(f"Corrected gap for radius {self.radius*1e3:.1f} nm: {self.gap*1e3:.2f} nm")

        # Right triangle tip
        P1 = np.array([
            self.center[0] + self.corected_gap/2.0,
            self.center[1]
        ])

        # Equilateral triangle assumption
        P2 = P1 + np.array([
            self.length,
            self.width/2.0
        ])

        P3 = P2 * np.array([1.0, -1.0])

        tip_right = [
            mp.Vector3(*P1),
            mp.Vector3(*P2),
            mp.Vector3(*P3)
        ]

        # Mirror on Y axis
        mirror = np.array([-1.0, 1.0])

        tip_left = [
            mp.Vector3(*(P1 * mirror)),
            mp.Vector3(*(P2 * mirror)),
            mp.Vector3(*(P3 * mirror))
        ]

        x_centroid = (P1[0] + P2[0] + P3[0]) / 3.0
        bow_tie = [
            mp.Prism(
                tip_right,
                height=self.thickness,
                material=self.material,
                center=mp.Vector3(x_centroid, 0, self.z_offset)
            ),
            mp.Prism(
                tip_left,
                height=self.thickness,
                material=self.material,
                center=mp.Vector3(-x_centroid, 0, self.z_offset)
            )
        ]

        # Fillets / rounding
        if self.radius > 1e-12:

            bow_tie += clear_edges_bowtie(
                points=[P1, P2, P3],
                antenna=self,
            )

            bow_tie += clear_edges_bowtie(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                antenna=self,
            )

            bow_tie += fillet_bowtie(
                points=[P1, P2, P3],
                antenna=self
            )

            bow_tie += fillet_bowtie(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                antenna=self
            )

        return bow_tie

    # -----------------------------------------------------

    def bounding_box(self):

        return [
            2*self.length + self.gap,
            self.width,
            self.thickness
        ]

# =========================================================
# SplitBar
# =========================================================

class SplitBar(AntennaBase):

    def __init__(self,
                 gap,
                 length,
                 width,
                 thickness,
                 material,
                 z_offset=0.0,
                 center=(0.0, 0.0),
                 radius=0.0):

        self.gap = gap
        self.length = length
        self.width = width
        self.thickness = thickness
        self.material = material
        self.z_offset = z_offset
        self.center = np.array(center)
        self.radius = radius

        if self.radius > self.width/2.0:
            print(f"Warning: radius {self.radius*1e3:.1f} nm is too large for width {self.width*1e3:.1f} nm. Consider reducing the radius to avoid geometry issues.")

    def build_geometry(self):

        split_bar = [
            mp.Block(
                # right bar
                mp.Vector3(self.length, self.width, self.thickness),
                center = mp.Vector3(self.center[0] + self.gap/2.0 + self.length/2.0,
                                    self.center[1],
                                    self.z_offset),
                material = self.material,
            ),
            
            mp.Block(
                mp.Vector3(self.length, self.width, self.thickness),
                center = mp.Vector3(self.center[0] - self.gap/2.0 - self.length/2.0,
                                    self.center[1],
                                    self.z_offset),
                material = self.material,
            )
        ]

        # Fillets / rounding
        if self.radius > 1e-12:
            P1 = np.array([self.center[0] + self.gap/2.0,
                  self.center[1] + self.width/2.0])
            P2 = np.array([self.center[0] + self.gap/2.0,
                  self.center[1] - self.width/2.0])
            P3 = np.array([self.center[0] + self.gap/2.0 + self.length,
                  self.center[1] + self.width/2.0])
            P4 = np.array([self.center[0] + self.gap/2.0 + self.length,
                  self.center[1] - self.width/2.0])
            mirror = np.array([-1.0, 1.0])

            split_bar += clear_rectangle_corners(
                points=[P1, P2, P3, P4],
                antenna=self
            )
            split_bar += clear_rectangle_corners(
                points=[P1 * mirror, P2 * mirror, P3 * mirror, P4 * mirror],
                antenna=self
            )
            split_bar += fillet_rectangle(
                points=[P1, P2, P3, P4],
                antenna=self
            )
            split_bar += fillet_rectangle(
                points=[P1 * mirror, P2 * mirror, P3 * mirror, P4 * mirror],
                antenna=self
            )

        return split_bar

    # -----------------------------------------------------

    def bounding_box(self):

        return [
            self.length + self.gap,
            self.width,
            self.thickness
        ]
    
# =========================================================
# SplitBar
# =========================================================

class Bar(AntennaBase):

    def __init__(self,
                 length,
                 width,
                 thickness,
                 material,
                 z_offset=0.0,
                 center=(0.0, 0.0),
                 radius=0.0):

        self.length = length
        self.width = width
        self.thickness = thickness
        self.material = material
        self.z_offset = z_offset
        self.center = np.array(center)
        self.radius = radius

        if self.radius > self.width/2.0:
            print(f"Warning: radius {self.radius*1e3:.1f} nm is too large for width {self.width*1e3:.1f} nm. Consider reducing the radius to avoid geometry issues.")

    def build_geometry(self):

        bar = [
            mp.Block(
                mp.Vector3(self.length, self.width, self.thickness),
                center = mp.Vector3(self.center[0],
                                    self.center[1],
                                    self.z_offset),
                material = self.material,
            )
        ]

        # Fillets / rounding
        if self.radius > 1e-12:
            P1 = np.array([self.center[0] + self.length/2.0,
                  self.center[1] + self.width/2.0])
            P2 = np.array([self.center[0] + self.length/2.0,
                  self.center[1] - self.width/2.0])
            P3 = np.array([self.center[0] - self.length/2.0,
                  self.center[1] + self.width/2.0])
            P4 = np.array([self.center[0] - self.length/2.0,
                  self.center[1] - self.width/2.0])

            bar += clear_rectangle_corners(
                points=[P1, P2, P3, P4],
                antenna=self
            )
            bar += fillet_rectangle(
                points=[P1, P2, P3, P4],
                antenna=self
            )

        return bar

    # -----------------------------------------------------

    def bounding_box(self):

        return [
            self.length,
            self.width,
            self.thickness
        ]
