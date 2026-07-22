"""
=============================================================================
FEATURE ENGINEER — 2D POSTURE ANALYSIS (REVISED)
=============================================================================
Fokus: koordinat x, y saja (2D image plane)
Label: forward_head (FHP), kyphosis, rounded_shoulder (RSP), normal

Perubahan utama dari versi sebelumnya:
  1. HIP sebagai anchor point utama untuk semua kondisi
     → ear_hip_x  : mendeteksi FHP bahkan saat RSP hadir bersamaan
     → shoulder_hip_x : mendeteksi RSP bahkan saat FHP hadir bersamaan
     → Tanpa ini, FHP+RSP bisa terdeteksi sebagai normal

  2. Segment ratio (ear–shoulder / shoulder–hip)
     → Membedakan kyphosis dari FHP+RSP
     → Kyphosis: segmen atas memendek, ratio mengecil

  3. Curvature index diperbaiki
     → Formula: (ear_shoulder_dist + shoulder_hip_dist) / ear_hip_dist
     → Normal: mendekati 1.0 (hampir lurus)
     → Kyphosis/kombinasi: > 1.0 (ada lekukan)

  4. Fitur RSP direvisi agar tidak overlap dengan FHP
     → Semua diukur relatif terhadap hip, bukan relatif terhadap ear

  5. Combination pattern detector (D6)
     → ear_hip_x_norm × shoulder_hip_x_norm
     → Membedakan FHP+RSP dari kondisi tunggal

Catatan:
  - Semua koordinat z diabaikan, hanya x dan y yang digunakan
  - Normalisasi menggunakan torso_length (shoulder_center → hip_center)
  - Semua sudut dalam derajat
  - Total tetap 54 fitur engineered (A=12, B=16, C=9, D=7, E=5, F=5)
=============================================================================
"""

import numpy as np
import cv2
import mediapipe as mp


class FeatureEngineer:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    # =========================================================
    # EKSTRAKSI LANDMARK DARI IMAGE
    # =========================================================

    def extract_landmarks(self, image: np.ndarray):
        """
        Ekstrak 33 pose landmarks dari gambar OpenCV (BGR).
        Return: np.array shape (132,) → 33 * 4 (x, y, z, visibility)
                None jika gagal deteksi
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=10)

        results = self.pose.process(image_rgb)

        if results.pose_landmarks is None:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        landmarks = np.array(landmarks, dtype=np.float32)

        if len(landmarks) != 132:
            return None

        return landmarks

    # =========================================================
    # UTILITAS
    # =========================================================

    def _angle_2d(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Hitung sudut ABC dengan B sebagai vertex, dalam derajat.
        Input: titik 2D masing-masing shape (2,)
        """
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cos_angle = np.clip(
            np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0
        )
        return float(np.degrees(np.arccos(cos_angle)))

    def _angle_with_vertical(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Hitung sudut vektor (p1 → p2) terhadap sumbu vertikal (0, 1).
        """
        vec = p2 - p1
        vertical = np.array([0.0, 1.0])
        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-6:
            return 0.0
        cos_angle = np.clip(
            np.dot(vec / norm_vec, vertical), -1.0, 1.0
        )
        return float(np.degrees(np.arccos(cos_angle)))

    def _safe_div(self, numerator: float, denominator: float,
                  fallback: float = 0.0) -> float:
        """Pembagian aman, return fallback jika denominator ~ 0."""
        if abs(denominator) < 1e-6:
            return fallback
        return numerator / denominator

    # =========================================================
    # MAIN: CALCULATE ENHANCED FEATURES
    # =========================================================

    def calculate_enhanced_features(self, landmarks) -> list:
        """
        Hitung 54 engineered features dari raw landmarks (132 nilai).
        Semua fitur berbasis 2D (x, y). Koordinat z diabaikan.

        Struktur:
          Blok A: FHP      — 12 fitur
          Blok B: Kyphosis — 16 fitur
          Blok C: RSP      —  9 fitur
          Blok D: Global   —  7 fitur
          Blok E: Quality  —  5 fitur
          Blok F: Normal   —  5 fitur
          Total            = 54 fitur

        Return: list of 54 float, atau None jika gagal
        """
        if landmarks is None:
            return None

        lm = np.array(landmarks, dtype=np.float32)
        if len(lm) != 132:
            return None

        # Reshape → (33, 4): kolom [x, y, z, visibility]
        pts = lm.reshape(33, 4)

        # Helper: ambil x, y saja dari index landmark
        def p(idx): return pts[idx, :2]

        try:
            # ---------------------------------------------------
            # LANDMARK UTAMA (2D)
            # ---------------------------------------------------
            nose = p(0)
            left_eye = p(2)
            right_eye = p(5)
            left_ear = p(7)
            right_ear = p(8)
            left_shoulder = p(11)
            right_shoulder = p(12)
            left_elbow = p(13)
            right_elbow = p(14)
            left_hip = p(23)
            right_hip = p(24)
            left_knee = p(25)
            right_knee = p(26)

            # ---------------------------------------------------
            # CENTER POINTS
            # ---------------------------------------------------
            ear_center = (left_ear + right_ear) / 2
            eye_center = (left_eye + right_eye) / 2
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            knee_center = (left_knee + right_knee) / 2
            elbow_center = (left_elbow + right_elbow) / 2

            # ---------------------------------------------------
            # NORMALISASI & JARAK DASAR
            # ---------------------------------------------------
            torso_length = np.linalg.norm(shoulder_center - hip_center)
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

            # Jarak antar titik utama (dipakai berulang di beberapa blok)
            ear_shoulder_dist = np.linalg.norm(ear_center - shoulder_center)
            shoulder_hip_dist = np.linalg.norm(shoulder_center - hip_center)
            ear_hip_dist = np.linalg.norm(ear_center - hip_center)

            features = []

            # =================================================
            # BLOK A: FORWARD HEAD POSTURE (FHP) — 12 fitur
            # =================================================
            # Proxy dari CVA (Craniovertebral Angle).
            # Titik kunci: ear sebagai representasi posisi kepala,
            # hip sebagai anchor yang tidak bergerak saat FHP.
            #
            # Mengapa butuh ear–hip (A7) bukan hanya ear–shoulder:
            #   Saat FHP+RSP, bahu ikut maju bersama kepala
            #   → ear–shoulder offset mengecil (keduanya sama-sama maju)
            #   → tapi ear–hip tetap besar karena hip diam
            #   → A7 mendeteksi FHP bahkan dalam kombinasi FHP+RSP

            # A1. Ear–shoulder horizontal offset (raw)
            ear_shoulder_x = shoulder_center[0] - ear_center[0]
            features.append(ear_shoulder_x)

            # A2. Ear–shoulder horizontal offset normalized
            features.append(self._safe_div(ear_shoulder_x, torso_length))

            # A3. Neck angle: sudut ear–shoulder–hip
            #     Normal ~170-180°, FHP < 170°
            neck_angle = self._angle_2d(
                ear_center, shoulder_center, hip_center
            )
            features.append(neck_angle)

            # A4. Deviasi neck_angle dari 180°
            neck_angle_dev = abs(180.0 - neck_angle)
            features.append(neck_angle_dev)

            # A5. Cranio-vertical angle
            #     Sudut vektor ear→shoulder terhadap vertikal
            #     Normal ~0°, FHP membesar
            cranio_vertical = self._angle_with_vertical(
                ear_center, shoulder_center
            )
            features.append(cranio_vertical)

            # A6. Nose–shoulder horizontal offset normalized
            nose_shoulder_x = shoulder_center[0] - nose[0]
            features.append(self._safe_div(nose_shoulder_x, torso_length))

            # A7. ★ EAR–HIP horizontal offset normalized (ANCHOR HIP)
            #     Fitur kritis untuk kombinasi FHP+RSP
            ear_hip_x = abs(ear_center[0] - hip_center[0])
            ear_hip_x_norm = self._safe_div(ear_hip_x, torso_length)
            features.append(ear_hip_x_norm)

            # A8. Sudut vektor hip→ear terhadap vertikal
            #     Normal: hampir 0° (ear tepat di atas hip)
            #     FHP   : membesar (ear maju dari hip)
            ear_hip_vertical = self._angle_with_vertical(
                hip_center, ear_center
            )
            features.append(ear_hip_vertical)

            # A9. Nose–shoulder–hip angle
            nose_shoulder_hip = self._angle_2d(
                nose, shoulder_center, hip_center
            )
            features.append(nose_shoulder_hip)

            # A10. Ear vertical position relatif shoulder
            ear_shoulder_y = ear_center[1] - shoulder_center[1]
            features.append(ear_shoulder_y)

            # A11. Ear simetri kiri–kanan (head tilt)
            ear_symmetry_y = abs(left_ear[1] - right_ear[1])
            features.append(ear_symmetry_y)

            # A12. FHP severity index (gabungan A2 + A7 + A4)
            fhp_severity = (
                abs(self._safe_div(ear_shoulder_x, torso_length)) +
                ear_hip_x_norm +
                neck_angle_dev / 180.0
            )
            features.append(fhp_severity)

            # =================================================
            # BLOK B: KYPHOSIS — 15 fitur
            # =================================================
            # Proxy dari Cobb Angle.
            # Kyphosis tidak terlihat langsung dari foto — yang dideteksi
            # adalah pola kompensasi: kepala+bahu maju DAN torso bengkok.
            #
            # Perbedaan FHP+RSP vs FHP+RSP+Kyphosis:
            #   FHP+RSP:          ear maju, shoulder maju, torso LURUS
            #                     → curvature ~1.0, segment ratio normal
            #   FHP+RSP+Kyphosis: ear maju, shoulder maju, torso BENGKOK
            #                     → curvature > 1.0, segment ratio mengecil
            #
            # B5 (curvature index) dan B6 (segment ratio) adalah
            # dua fitur utama pembeda kyphosis dari FHP+RSP murni.

            # B1. 3-Point angle: ear–shoulder–hip
            three_point_angle = self._angle_2d(
                ear_center, shoulder_center, hip_center
            )
            features.append(three_point_angle)

            # B2. Deviasi dari 180°
            three_point_dev = abs(180.0 - three_point_angle)
            features.append(three_point_dev)

            # B3. Shoulder–hip horizontal deviation (raw)
            shoulder_hip_x = shoulder_center[0] - hip_center[0]
            features.append(shoulder_hip_x)

            # B4. Shoulder–hip horizontal deviation normalized
            shoulder_hip_x_norm = self._safe_div(
                shoulder_hip_x, torso_length
            )
            features.append(shoulder_hip_x_norm)

            # B5. ★ CURVATURE INDEX (formula yang benar)
            #     (ear_shoulder + shoulder_hip) / ear_hip
            #     Jika lurus sempurna: rasio = 1.0
            #     Jika ada lekukan (kyphosis): rasio > 1.0
            #     Makin besar kyphosis, makin besar nilainya
            curvature_index = self._safe_div(
                ear_shoulder_dist + shoulder_hip_dist,
                ear_hip_dist
            )
            features.append(curvature_index)

            # B6. ★ SEGMENT RATIO (ear–shoulder / shoulder–hip)
            #     Kyphosis → leher ikut membungkuk, jarak ear–shoulder memendek
            #     FHP murni → leher tegak, ear–shoulder lebih panjang relatif
            #     Nilai kecil → indikasi kyphosis ada
            segment_ratio = self._safe_div(
                ear_shoulder_dist, shoulder_hip_dist
            )
            features.append(segment_ratio)

            # B7. Sudut shoulder–hip terhadap vertikal
            shoulder_hip_vertical = self._angle_with_vertical(
                hip_center, shoulder_center
            )
            features.append(shoulder_hip_vertical)

            # B8. Sudut ear–shoulder terhadap vertikal
            ear_shoulder_vertical = self._angle_with_vertical(
                shoulder_center, ear_center
            )
            features.append(ear_shoulder_vertical)

            # B9. Upper body compactness
            upper_body_ratio = self._safe_div(
                ear_shoulder_dist, torso_length
            )
            features.append(upper_body_ratio)

            # B10. Elbow compensation angle
            #      Kyphosis → siku ikut maju ke depan
            elbow_shoulder_vertical = self._angle_with_vertical(
                shoulder_center, elbow_center
            )
            features.append(elbow_shoulder_vertical)

            # B11. Shoulder height asymmetry
            shoulder_height_asym = abs(
                left_shoulder[1] - right_shoulder[1]
            )
            features.append(shoulder_height_asym)

            # B12. Kyphosis severity index
            kypho_severity = (
                three_point_dev / 180.0 +
                abs(shoulder_hip_x_norm) +
                max(0.0, curvature_index - 1.0)
            )
            features.append(kypho_severity)

            # B_new_1: Ear-hip vertical distance normalized
            # Kyphosis mempersingkat tinggi tubuh → nilai ini mengecil
            ear_hip_vertical = abs(ear_center[1] - hip_center[1])
            ear_hip_vertical_norm = self._safe_div(
                ear_hip_vertical, torso_length)
            features.append(ear_hip_vertical_norm)

            # B_new_2: Elbow-hip horizontal offset normalized
            # Kyphosis → seluruh upper body condong, siku maju lebih jauh dari hip
            elbow_hip_x = abs(elbow_center[0] - hip_center[0])
            elbow_hip_x_norm = self._safe_div(elbow_hip_x, torso_length)
            features.append(elbow_hip_x_norm)

            # B_new_3: Vertical compression ratio
            # Rasio jarak vertikal vs diagonal ear-hip
            # Kyphosis mengubah rasio ini secara unik
            ear_hip_total = np.linalg.norm(ear_center - hip_center)
            vertical_compression = self._safe_div(
                ear_hip_vertical, ear_hip_total)
            features.append(vertical_compression)

            # B_new_4: Perpendicular distance siku dari garis bahu-panggul
            # ★ FITUR KUNCI untuk membedakan FHP+RSP dari FHP+Kyphosis
            #
            # Logika:
            #   FHP+RSP murni: torso lurus → siku dekat garis bahu-panggul
            #                  → perp_dist_norm KECIL
            #   FHP+Kyphosis:  torso bengkok → siku terdorong keluar
            #                  → perp_dist_norm BESAR
            #
            # Ini berbeda dari B10 (elbow_compensation angle) karena:
            #   B10 mengukur SUDUT siku terhadap vertikal
            #   B_new_4 mengukur JARAK TEGAK LURUS siku dari garis bahu-panggul
            #   Keduanya menangkap aspek berbeda dari kompensasi siku
            line_vec = hip_center - shoulder_center
            line_len = np.linalg.norm(line_vec)
            if line_len > 1e-6:
                line_unit = line_vec / line_len
                elbow_vec = elbow_center - shoulder_center
                projection = np.dot(elbow_vec, line_unit)
                projected_point = shoulder_center + projection * line_unit
                perp_dist = np.linalg.norm(elbow_center - projected_point)
                perp_dist_norm = self._safe_div(perp_dist, torso_length)
            else:
                perp_dist_norm = 0.0
            features.append(perp_dist_norm)

            # =================================================
            # BLOK C: ROUNDED SHOULDER (RSP) — 9 fitur
            # =================================================
            # Proxy dari FSA (Forward Shoulder Angle).
            # SEMUA fitur RSP diukur relatif terhadap HIP sebagai anchor,
            # bukan relatif terhadap ear.
            #
            # Mengapa penting:
            #   Saat FHP+RSP, ear–shoulder offset kecil (keduanya maju)
            #   tapi shoulder–hip offset TETAP BESAR karena hip diam
            #   → C1 mendeteksi RSP bahkan dalam kombinasi FHP+RSP

            # C1. ★ SHOULDER–HIP horizontal offset normalized (ANCHOR HIP)
            rsp_shoulder_hip_x = abs(shoulder_center[0] - hip_center[0])
            rsp_shoulder_hip_norm = self._safe_div(
                rsp_shoulder_hip_x, torso_length
            )
            features.append(rsp_shoulder_hip_norm)

            # C2. FSA Proxy: sudut vektor hip→shoulder terhadap vertikal
            #     Normal: kecil (torso tegak)
            #     RSP   : membesar (bahu maju dari hip)
            fsa_proxy = self._angle_with_vertical(
                hip_center, shoulder_center
            )
            features.append(fsa_proxy)

            # C3. Left shoulder–hip offset normalized
            left_sh_hip_x = abs(left_shoulder[0] - left_hip[0])
            features.append(self._safe_div(left_sh_hip_x, torso_length))

            # C4. Right shoulder–hip offset normalized
            right_sh_hip_x = abs(right_shoulder[0] - right_hip[0])
            features.append(self._safe_div(right_sh_hip_x, torso_length))

            # C5. Bilateral RSP asymmetry (selisih kiri–kanan)
            rsp_bilateral = abs(
                self._safe_div(left_sh_hip_x, torso_length) -
                self._safe_div(right_sh_hip_x, torso_length)
            )
            features.append(rsp_bilateral)

            # C6. Shoulder–ear vertical proximity normalized
            #     RSP → bahu naik sedikit mendekati telinga
            shoulder_ear_y = abs(shoulder_center[1] - ear_center[1])
            features.append(self._safe_div(shoulder_ear_y, torso_length))

            # C7. Shoulder width normalized
            features.append(self._safe_div(shoulder_width, torso_length))

            # C8. Shoulder–hip–knee angle
            #     Melihat alignment bahu terhadap struktur bawah tubuh
            shoulder_hip_knee = self._angle_2d(
                shoulder_center, hip_center, knee_center
            )
            features.append(shoulder_hip_knee)

            # C9. RSP severity index
            rsp_severity = (
                rsp_shoulder_hip_norm +
                fsa_proxy / 90.0 +
                rsp_bilateral
            )
            features.append(rsp_severity)

            # =================================================
            # BLOK D: GLOBAL ALIGNMENT — 7 fitur
            # =================================================
            # Fitur lintas kondisi yang membantu model memahami
            # postur secara keseluruhan, termasuk pola kombinasi.

            # D1. Plumb line: ear–hip horizontal offset normalized
            plumb_line_x = abs(ear_center[0] - hip_center[0])
            features.append(self._safe_div(plumb_line_x, torso_length))

            # D2. Overall body lean (sudut hip→ear terhadap vertikal)
            body_lean = self._angle_with_vertical(hip_center, ear_center)
            features.append(body_lean)

            # D3. Head–torso ratio
            head_torso_ratio = self._safe_div(
                ear_shoulder_dist, torso_length
            )
            features.append(head_torso_ratio)

            # D4. Center of mass deviation
            com_x = (
                ear_center[0] + shoulder_center[0] + hip_center[0]
            ) / 3
            com_deviation = self._safe_div(
                abs(com_x - hip_center[0]), torso_length
            )
            features.append(com_deviation)

            # D5. Lateral symmetry bahu
            lateral_sym = self._safe_div(
                abs(left_shoulder[1] - right_shoulder[1]), torso_length
            )
            features.append(lateral_sym)

            # D6. ★ COMBINATION PATTERN DETECTOR
            #     Produk ear_hip_x_norm × shoulder_hip_x_norm
            #     Pola unik tiap kondisi:
            #       Normal  : kecil × kecil = sangat kecil
            #       FHP saja: besar × kecil = kecil-sedang
            #       RSP saja: kecil × besar = kecil-sedang
            #       FHP+RSP : besar × besar = BESAR (nilai unik!)
            combination_pattern = ear_hip_x_norm * rsp_shoulder_hip_norm
            features.append(combination_pattern)

            # D7. Overall postural deviation index
            postural_deviation = np.sqrt(
                fhp_severity**2 + kypho_severity**2 + rsp_severity**2
            )
            features.append(postural_deviation)

            # =================================================
            # BLOK E: VISIBILITY / QUALITY — 5 fitur
            # =================================================
            key_landmark_idx = [0, 7, 8, 11, 12, 23, 24]
            vis_scores = [pts[i, 3] for i in key_landmark_idx]

            # E1. Rata-rata visibility landmark kunci
            features.append(float(np.mean(vis_scores)))

            # E2. Minimum visibility
            features.append(float(np.min(vis_scores)))

            # E3. Standar deviasi visibility
            features.append(float(np.std(vis_scores)))

            # E4. Bilateral confidence (selisih visibility kiri–kanan)
            left_vis = np.mean([pts[7, 3], pts[11, 3], pts[23, 3]])
            right_vis = np.mean([pts[8, 3], pts[12, 3], pts[24, 3]])
            bilateral_conf = 1.0 - abs(left_vis - right_vis)
            features.append(float(bilateral_conf))

            # E5. Overall pose quality
            features.append(float(np.mean(vis_scores)) * bilateral_conf)

            # =================================================
            # BLOK F: NORMAL — 5 fitur
            # =================================================

            # F1. Ear-hip alignment score (mendekati 0 = normal)
            norm_ear_hip = 1.0 - min(ear_hip_x_norm, 1.0)
            features.append(norm_ear_hip)

            # F2. Shoulder-hip alignment score
            norm_sh_hip = 1.0 - min(rsp_shoulder_hip_norm, 1.0)
            features.append(norm_sh_hip)

            # F3. Curvature proximity to 1.0
            norm_curvature = 1.0 - abs(curvature_index - 1.0)
            features.append(max(norm_curvature, 0.0))

            # F4. Overall alignment composite
            norm_composite = (norm_ear_hip + norm_sh_hip + max(norm_curvature, 0.0)) / 3.0
            features.append(norm_composite)

            # F5. Bilateral symmetry score
            norm_symmetry = 1.0 - lateral_sym
            features.append(max(norm_symmetry, 0.0))

        except Exception as e:
            print(f"  [FeatureEngineer] Error: {e}")
            return None

        # Blok A=12, B=16, C=9, D=7, E=5 F=5 → total=54
        assert len(features) == 54, \
            f"Jumlah fitur tidak sesuai: {len(features)} (expected 54)"

        return features

    # =========================================================
    # NAMA FITUR (untuk feature importance / interpretasi)
    # =========================================================

    def get_feature_names(self) -> list:
        """Return daftar nama 177 fitur (132 raw + 48 engineered)."""

        landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        assert len(landmark_names) == 33

        raw_names = []
        for name in landmark_names:
            raw_names += [f'{name}_x', f'{name}_y',
                          f'{name}_z', f'{name}_vis']

        engineered_names = [
            # FHP (12)
            'fhp_ear_shoulder_x',
            'fhp_ear_shoulder_x_norm',
            'fhp_neck_angle',
            'fhp_neck_angle_dev180',
            'fhp_cranio_vertical_angle',
            'fhp_nose_shoulder_x_norm',
            'fhp_ear_hip_x_norm',            # ★ anchor hip
            'fhp_ear_hip_vertical',           # ★ anchor hip
            'fhp_nose_shoulder_hip_angle',
            'fhp_ear_shoulder_y',
            'fhp_ear_symmetry_y',
            'fhp_severity_index',

            # Kyphosis (16)
            'kypho_3point_angle',
            'kypho_3point_dev180',
            'kypho_shoulder_hip_x',
            'kypho_shoulder_hip_x_norm',
            'kypho_curvature_index',          # ★ formula baru
            'kypho_segment_ratio',            # ★ baru
            'kypho_shoulder_hip_vertical',
            'kypho_ear_shoulder_vertical',
            'kypho_upper_body_ratio',
            'kypho_elbow_compensation',
            'kypho_shoulder_height_asym',
            'kypho_severity_index',
            'ear_hip_vertical_norm',
            'elbow_hip_x_norm',
            'vertical_compression',
            'kypho_elbow_perp_dist',

            # RSP (9)
            'rsp_shoulder_hip_x_norm',        # ★ anchor hip
            'rsp_fsa_proxy',                  # ★ FSA proxy
            'rsp_left_shoulder_hip_norm',
            'rsp_right_shoulder_hip_norm',
            'rsp_bilateral_asym',
            'rsp_shoulder_ear_y_norm',
            'rsp_shoulder_width_norm',
            'rsp_shoulder_hip_knee_angle',
            'rsp_severity_index',

            # Global (7)
            'global_plumb_line_norm',
            'global_body_lean',
            'global_head_torso_ratio',
            'global_com_deviation',
            'global_lateral_symmetry',
            'global_combination_pattern',     # ★ baru
            'global_postural_deviation',

            # Quality (5)
            'qual_avg_visibility',
            'qual_min_visibility',
            'qual_std_visibility',
            'qual_bilateral_confidence',
            'qual_pose_quality',

            # Normal (5)
            'norm_ear_hip',
            'norm_sh_hip',
            'norm_curvature',
            'norm_composite',
            'norm_symmetry'
        ]

        assert len(engineered_names) == 53, \
            f"Nama fitur tidak sesuai: {len(engineered_names)}"

        return raw_names + engineered_names
