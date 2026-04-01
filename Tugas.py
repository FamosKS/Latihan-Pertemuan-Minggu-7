import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import pywt 

def tugas_frekuensi_wavelet():
    print("=" * 70)
    print("TUGAS: TRANSFORMASI FOURIER, FILTERING FREKUENSI, DAN WAVELET")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # 1. PERSIAPAN CITRA
    # ---------------------------------------------------------
    def generate_test_images():
        images = {}
        
        # A. Citra "Natural" 
        img_nat = np.zeros((256, 256), dtype=np.float32)
        # Background gradient
        for i in range(256):
            img_nat[i, :] = np.linspace(50, 200, 256)
        cv2.rectangle(img_nat, (60, 60), (120, 180), 255, -1)
        cv2.circle(img_nat, (180, 100), 40, 30, -1)
        texture = np.random.normal(0, 15, (256, 256))
        img_nat = np.clip(img_nat + texture, 0, 255).astype(np.uint8)
        images['Natural (Tepi & Tekstur)'] = img_nat
        
        # B. Citra dengan Noise Periodik
        img_base = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(img_base, (128, 128), 60, 200, -1)
        x = np.arange(256)
        y = np.arange(256)
        X, Y = np.meshgrid(x, y)
        periodic_noise = 50 * np.sin(2 * np.pi * X / 16) + 50 * np.sin(2 * np.pi * Y / 16)
        img_periodic = np.clip(img_base + periodic_noise + 128, 0, 255).astype(np.uint8)
        images['Periodic Noise'] = img_periodic
        
        return images

    test_images = generate_test_images()

    # ---------------------------------------------------------
    # 2. TRANSFORMASI FOURIER & REKONSTRUKSI FASE/MAGNITUDO
    # ---------------------------------------------------------
    def analyze_and_reconstruct(image, title):
        # FFT
        img_float = image.astype(np.float32)
        f = fft2(img_float)
        fshift = fftshift(f)
        
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        log_mag = np.log(1 + magnitude)
        
        complex_phase_only = 1 * np.exp(1j * phase)
        recon_phase = np.abs(ifft2(ifftshift(complex_phase_only)))
        recon_phase = cv2.normalize(recon_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        complex_mag_only = magnitude * np.exp(1j * 0)
        recon_mag = np.abs(ifft2(ifftshift(complex_mag_only)))
        recon_mag = cv2.normalize(np.log(1 + recon_mag), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Visualisasi
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'{title}\nOriginal')
        axes[1].imshow(log_mag, cmap='gray')
        axes[1].set_title('Magnitude Spectrum (Log)')
        axes[2].imshow(recon_phase, cmap='gray')
        axes[2].set_title('Recon: Phase Only\n(Edge/Structure visible)')
        axes[3].imshow(recon_mag, cmap='gray')
        axes[3].set_title('Recon: Magnitude Only\n(Intensity/No Structure)')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()
        return fshift

    print("\n--- A. Analisis Spektrum & Rekonstruksi Komponen ---")
    fshift_nat = analyze_and_reconstruct(test_images['Natural (Tepi & Tekstur)'], 'Natural Image')
    fshift_per = analyze_and_reconstruct(test_images['Periodic Noise'], 'Periodic Noise')

    # ---------------------------------------------------------
    # 3. FILTERING DI DOMAIN FREKUENSI
    # ---------------------------------------------------------
    def frequency_filtering(image, fshift):
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        # Meshgrid untuk jarak D(u,v)
        u = np.arange(rows)
        v = np.arange(cols)
        U, V = np.meshgrid(v, u)
        D = np.sqrt((U - ccol)**2 + (V - crow)**2)
        
        # A. Ideal Lowpass Filter (Cutoff = 30)
        D0 = 30
        H_ideal_lpf = (D <= D0).astype(float)
        filtered_ideal = np.abs(ifft2(ifftshift(fshift * H_ideal_lpf)))
        
        # B. Gaussian Lowpass Filter (Cutoff = 30)
        H_gauss_lpf = np.exp(-(D**2) / (2 * (D0**2)))
        filtered_gauss = np.abs(ifft2(ifftshift(fshift * H_gauss_lpf)))
        
        # Visualisasi perbandingan Ringing Effect
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(filtered_ideal, cmap='gray')
        axes[1].set_title('Ideal LPF (Notice Ringing)')
        axes[2].imshow(filtered_gauss, cmap='gray')
        axes[2].set_title('Gaussian LPF (Smooth, No Ringing)')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    def notch_filter_periodic(image, fshift):
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        H_notch = np.ones((rows, cols), dtype=np.float32)
        r = 16 
        thickness = 4
       
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        D = np.sqrt((U - ccol)**2 + (V - crow)**2)
        H_notch[(D >= r - thickness) & (D <= r + thickness)] = 0
        
        filtered_notch = np.abs(ifft2(ifftshift(fshift * H_notch)))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Periodic Noise')
        axes[1].imshow(H_notch, cmap='gray')
        axes[1].set_title('Notch / Bandstop Filter Mask')
        axes[2].imshow(filtered_notch, cmap='gray')
        axes[2].set_title('Restored (Noise Removed)')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    print("\n--- B. Filtering Spektrum ---")
    frequency_filtering(test_images['Natural (Tepi & Tekstur)'], fshift_nat)
    notch_filter_periodic(test_images['Periodic Noise'], fshift_per)

    # ---------------------------------------------------------
    # 4. TRANSFORMASI WAVELET (Dekomposisi 2 Level)
    # ---------------------------------------------------------
    def wavelet_analysis(image):
        wavelet_type = 'db4'
        coeffs2 = pywt.wavedec2(image, wavelet_type, level=2)
        LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs2
        
        # Plot koefisien (Level 1)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle('Wavelet Decomposition (Level 1 Detail)', fontsize=14)
        axes[0, 0].imshow(LL2, cmap='gray')
        axes[0, 0].set_title('Approximation (LL)')
        axes[0, 1].imshow(LH1, cmap='gray')
        axes[0, 1].set_title('Horizontal Detail (LH)')
        axes[1, 0].imshow(HL1, cmap='gray')
        axes[1, 0].set_title('Vertical Detail (HL)')
        axes[1, 1].imshow(HH1, cmap='gray')
        axes[1, 1].set_title('Diagonal Detail (HH)')
        for ax in axes.ravel(): ax.axis('off')
        plt.tight_layout()
        plt.show()

        coeffs_recon = [LL2, (LH2, HL2, HH2), (np.zeros_like(LH1), np.zeros_like(HL1), np.zeros_like(HH1))]
        img_recon = pywt.waverec2(coeffs_recon, wavelet_type)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(img_recon, cmap='gray')
        axes[1].set_title('Wavelet Recon\n(Level 1 Details Removed / Denoised)')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    print("\n--- C. Transformasi Wavelet ---")
    wavelet_analysis(test_images['Natural (Tepi & Tekstur)'])

tugas_frekuensi_wavelet()