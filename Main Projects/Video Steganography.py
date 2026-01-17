import os
import struct
import secrets
import traceback
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SYNC = b'VSTG'
SYNC_REPEAT = 32
BIT_REP = 3
MAX_VERIFY_BYTES = 200000

NEON = "#1E90FF"
WHITE = "#FFFFFF"
LIGHT_CYAN = "#DFF7FF"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 600

def derive_key(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
    return kdf.derive(password.encode('utf-8'))

def encrypt_message(password: str, plaintext: str) -> bytes:
    salt = secrets.token_bytes(16)
    key = derive_key(password, salt)
    aes = AESGCM(key)
    nonce = secrets.token_bytes(12)
    ct = aes.encrypt(nonce, plaintext.encode('utf-8'), None)
    header = b'VSTG' + salt + nonce + struct.pack('>I', len(ct))
    return header + ct

def decrypt_message(password: str, payload: bytes) -> str:
    if len(payload) < 36:
        raise ValueError("Payload too short.")
    if payload[:4] != b'VSTG':
        raise ValueError("Invalid header magic.")
    salt = payload[4:20]
    nonce = payload[20:32]
    ct_len = struct.unpack('>I', payload[32:36])[0]
    ct = payload[36:36 + ct_len]
    if len(ct) != ct_len:
        raise ValueError("Ciphertext truncated.")
    key = derive_key(password, salt)
    aes = AESGCM(key)
    pt = aes.decrypt(nonce, ct, None)
    return pt.decode('utf-8')

def bytes_to_bits_redundant(data: bytes, rep: int = BIT_REP) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    bits_rep = np.repeat(bits, rep)
    return bits_rep.astype(np.uint8)

def majority_decode_bits(bits_rep: np.ndarray, rep: int = BIT_REP) -> bytes:
    if bits_rep.size % rep != 0:
        raise ValueError("Repeated bits length invalid.")
    groups = bits_rep.reshape((-1, rep))
    votes = (np.sum(groups, axis=1) > (rep // 2)).astype(np.uint8)
    return np.packbits(votes).tobytes()

def read_video_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise IOError("Video contains no readable frames.")
    return frames, fps, width, height

def write_video_frames_ffv1(path: str, frames: list, fps: float, width: int, height: int):
    if os.path.splitext(path)[1] == '':
        path += '.mkv'
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            "VideoWriter failed to open with FFV1. Your OpenCV may lack FFmpeg/FFV1 support.\n"
            "Options:\n"
            "  1) Install an ffmpeg-backed OpenCV wheel, or build OpenCV with ffmpeg support.\n"
            "  2) Use system ffmpeg to create an mkv with ffv1 from PNG frames:\n"
            "     ffmpeg -framerate <fps> -i frames/frame_%06d.png -c:v ffv1 output.mkv\n"
        )
    for f in frames:
        writer.write(f)
    writer.release()

def embed_bits_into_frames(frames: list, bits: np.ndarray) -> list:
    total_capacity = sum((f.shape[0] * f.shape[1]) for f in frames)
    if bits.size > total_capacity:
        raise ValueError(f"Insufficient capacity: need {bits.size}, have {total_capacity}")
    idx = 0
    out = []
    for f in frames:
        img = f.copy()
        h, w = img.shape[:2]
        blue = img[:, :, 0].flatten()
        to_write = min(blue.size, bits.size - idx)
        if to_write > 0:
            blue[:to_write] = (blue[:to_write] & 0xFE) | bits[idx:idx + to_write]
            idx += to_write
            img[:, :, 0] = blue.reshape((h, w))
        out.append(img)
    if idx < bits.size:
        raise ValueError("Ran out of frames while embedding.")
    return out

def extract_bits_from_frames(frames: list, n_bits: int) -> np.ndarray:
    bits = np.zeros(n_bits, dtype=np.uint8)
    idx = 0
    for f in frames:
        blue = f[:, :, 0].flatten()
        take = min(blue.size, n_bits - idx)
        if take > 0:
            bits[idx:idx + take] = blue[:take] & 1
            idx += take
        if idx >= n_bits:
            break
    if idx < n_bits:
        raise ValueError("Not enough bits available to extract.")
    return bits

def make_payload_bits(password: str, secret: str) -> (np.ndarray, bytes):
    enc = encrypt_message(password, secret)
    sync_block = SYNC * SYNC_REPEAT
    raw = sync_block + enc
    bits_rep = bytes_to_bits_redundant(raw, BIT_REP)
    return bits_rep, raw

def find_payload_in_frames(frames: list) -> bytes:
    h, w = frames[0].shape[:2]
    total_bits = len(frames) * h * w
    max_scan_bytes = min((total_bits // BIT_REP) // 8, MAX_VERIFY_BYTES)
    scan_bits = max_scan_bytes * 8 * BIT_REP
    scan_bits = min(scan_bits, total_bits)
    bits = extract_bits_from_frames(frames, scan_bits)
    try:
        raw = majority_decode_bits(bits, BIT_REP)
    except Exception as e:
        raise ValueError(f"Majority decode failed during scan: {e}")
    sync_block = SYNC * SYNC_REPEAT
    idx = raw.find(sync_block)
    if idx == -1:
        half = SYNC * (SYNC_REPEAT // 2)
        idx2 = raw.find(half)
        if idx2 == -1:
            raise ValueError("Magic header VSTG not found in bitstream.")
        idx = idx2
    payload_start = idx + len(sync_block)
    if payload_start + 36 > len(raw):
        bits_full = extract_bits_from_frames(frames, total_bits)
        raw_full = majority_decode_bits(bits_full, BIT_REP)
        idx = raw_full.find(sync_block)
        if idx == -1:
            raise ValueError("Magic header not found after full extraction.")
        payload_start = idx + len(sync_block)
        raw = raw_full
    header = raw[payload_start:payload_start + 36]
    if header[:4] != SYNC:
        raise ValueError("Header missing after sync block.")
    ct_len = struct.unpack('>I', header[32:36])[0]
    total_needed = payload_start + 36 + ct_len
    if total_needed > len(raw):
        bits_full = extract_bits_from_frames(frames, total_bits)
        raw_full = majority_decode_bits(bits_full, BIT_REP)
        idx = raw_full.find(sync_block)
        if idx == -1:
            raise ValueError("Magic header not found after full extraction.")
        payload_start = idx + len(sync_block)
        header = raw_full[payload_start:payload_start + 36]
        ct_len = struct.unpack('>I', header[32:36])[0]
        total_needed = payload_start + 36 + ct_len
        if total_needed > len(raw_full):
            raise ValueError("Stego data truncated in the video.")
        payload = raw_full[payload_start:total_needed]
        return payload
    payload = raw[payload_start:total_needed]
    return payload

def style_capsule_button(btn):
    btn.configure(bg=WHITE, fg="#000000", activebackground="#F0F0F0",
                  relief="raised", bd=1, padx=18, pady=8, font=("Segoe UI Semibold", 11))
    def on_enter(e):
        e.widget.configure(bg="#F7F7F7")
    def on_leave(e):
        e.widget.configure(bg=WHITE)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

def style_secondary_button(btn):
    btn.configure(bg="#BEEAFF", fg="#000000", activebackground="#99E0FF",
                  relief="raised", bd=1, padx=12, pady=6, font=("Segoe UI", 10))
    def on_enter(e):
        e.widget.configure(bg="#99E0FF")
    def on_leave(e):
        e.widget.configure(bg="#BEEAFF")
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

class VideoStegoApp:
    def __init__(self, root):
        self.root = root
        root.title("LSB Video Steganography — Premium")
        root.geometry("760x420")
        root.resizable(False, False)
        root.configure(bg=LIGHT_CYAN)
        try:
            root.attributes('-alpha', 0.95)
        except Exception:
            pass
        self.build_home()

    def build_home(self):
        frame = tk.Frame(self.root, bg=LIGHT_CYAN)
        frame.pack(expand=True, fill='both')

        title = tk.Label(frame, text="LSB Video Steganography", font=("Segoe UI", 20, "bold"), fg=NEON, bg=LIGHT_CYAN)
        subtitle = tk.Label(frame, text="Password-encrypted • AES-GCM • Save as MKV (FFV1) for lossless embedding", font=("Segoe UI", 10), fg="#054A66", bg=LIGHT_CYAN)
        title.pack(pady=(60, 8))
        subtitle.pack()

        btn_frame = tk.Frame(frame, bg=LIGHT_CYAN)
        btn_frame.pack(pady=20)

        embed_btn = tk.Button(btn_frame, text="Embedding", width=18, command=self.open_embedding_window)
        extract_btn = tk.Button(btn_frame, text="Extraction", width=18, command=self.open_extraction_window)
        style_capsule_button(embed_btn)
        style_capsule_button(extract_btn)
        embed_btn.grid(row=0, column=0, padx=12, pady=8)
        extract_btn.grid(row=0, column=1, padx=12, pady=8)

        footer = tk.Label(frame, text="Tip: Save as MKV (FFV1) for guaranteed LSB preservation.", bg=LIGHT_CYAN, fg="#05607A", font=("Segoe UI", 9))
        footer.pack(side='bottom', pady=16)

    def open_embedding_window(self):
        win = tk.Toplevel(self.root)
        win.title("Encryption — Embed")
        win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        win.resizable(False, False)
        win.configure(bg=LIGHT_CYAN)
        try:
            win.attributes('-alpha', 0.95)
        except Exception:
            pass

        menu = tk.Frame(win, bg=LIGHT_CYAN, width=200)
        menu.place(x=14, y=14, width=200, height=WINDOW_HEIGHT - 28)
        mlabel = tk.Label(menu, text="Menu", font=("Segoe UI Semibold", 12), fg=NEON, bg=LIGHT_CYAN)
        mlabel.pack(pady=(16,8))

        btn_upload = tk.Button(menu, text="Upload Video", width=16, command=lambda: self.upload_video(win))
        btn_embed = tk.Button(menu, text="Embed", width=16, command=lambda: self.embed_action(win))
        btn_clear = tk.Button(menu, text="Clear All", width=16, command=lambda: self.clear_embedding(win))
        btn_exit = tk.Button(menu, text="Exit", width=16, command=win.destroy)
        for b in (btn_upload, btn_embed, btn_clear, btn_exit):
            b.pack(pady=8)
            style_secondary_button(b)

        preview_frame = tk.Frame(win, bg="#E6FBFF", relief='sunken', bd=1)
        preview_frame.place(x=234, y=14, width=520, height=420)
        preview_label = tk.Label(preview_frame, text="No video uploaded", bg="#E6FBFF")
        preview_label.pack(expand=True)

        right = tk.Frame(win, bg=LIGHT_CYAN)
        right.place(x=770, y=14, width=320, height=420)
        s_lbl = tk.Label(right, text="Enter Secret Text", font=("Segoe UI", 10, "bold"), bg=LIGHT_CYAN)
        s_lbl.pack(anchor='nw', padx=10, pady=(10,4))
        secret_txt = scrolledtext.ScrolledText(right, wrap=tk.WORD, height=6)
        secret_txt.pack(fill='x', padx=10)

        p_lbl = tk.Label(right, text="Password (for encryption)", bg=LIGHT_CYAN)
        p_lbl.pack(anchor='nw', padx=10, pady=(8,2))
        pwd_entry = tk.Entry(right, show="*")
        pwd_entry.pack(fill='x', padx=10)

        params_frame = tk.LabelFrame(right, text="Output Parameters", bg=LIGHT_CYAN)
        params_frame.pack(fill='both', padx=10, pady=8)
        rows_var = tk.StringVar(master=win, value="Rows: -")
        cols_var = tk.StringVar(master=win, value="Cols: -")
        frames_var = tk.StringVar(master=win, value="Frames: -")
        fps_var = tk.StringVar(master=win, value="FPS: -")
        cap_var = tk.StringVar(master=win, value="Capacity (bits): -")
        tk.Label(params_frame, textvariable=rows_var, anchor='w', bg=LIGHT_CYAN).pack(fill='x')
        tk.Label(params_frame, textvariable=cols_var, anchor='w', bg=LIGHT_CYAN).pack(fill='x')
        tk.Label(params_frame, textvariable=frames_var, anchor='w', bg=LIGHT_CYAN).pack(fill='x')
        tk.Label(params_frame, textvariable=fps_var, anchor='w', bg=LIGHT_CYAN).pack(fill='x')
        tk.Label(params_frame, textvariable=cap_var, anchor='w', bg=LIGHT_CYAN).pack(fill='x')

        save_btn = tk.Button(right, text="Save Stego Video (.mkv FFV1)", command=lambda: self.save_stego_video(win))
        save_btn.pack(padx=10, pady=6, anchor='se')
        style_capsule_button(save_btn)

        log_frame = tk.LabelFrame(right, text="Log", bg=LIGHT_CYAN)
        log_frame.pack(fill='both', padx=10, pady=(6,12), expand=True)
        log_txt = scrolledtext.ScrolledText(log_frame, height=6)
        log_txt.pack(fill='both', expand=True)

        win.preview_label = preview_label
        win.secret_txt = secret_txt
        win.pwd_entry = pwd_entry
        win.rows_var = rows_var
        win.cols_var = cols_var
        win.frames_var = frames_var
        win.fps_var = fps_var
        win.cap_var = cap_var
        win.log_txt = log_txt
        win.uploaded_frames = None
        win.video_meta = None
        win.stego_frames = None
        win.raw_preview = None

    def log_embed(self, win, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        print(line, end='')
        try:
            win.log_txt.insert(tk.END, line)
            win.log_txt.see(tk.END)
        except Exception:
            pass

    def upload_video(self, win):
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")])
        if not path:
            return
        try:
            frames, fps, width, height = read_video_frames(path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read video: {e}")
            return
        win.uploaded_frames = frames
        win.video_meta = dict(fps=fps, width=width, height=height, path=path)
        frame_bgr = frames[0]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((500, 320))
        imgtk = ImageTk.PhotoImage(img)
        win.preview_label.configure(image=imgtk, text="")
        win.preview_label.image = imgtk
        h, w = frame_bgr.shape[:2]
        nframes = len(frames)
        cap = nframes * h * w
        win.rows_var.set(f"Rows: {h}")
        win.cols_var.set(f"Cols: {w}")
        win.frames_var.set(f"Frames: {nframes}")
        win.fps_var.set(f"FPS: {fps:.2f}")
        win.cap_var.set(f"Capacity (bits): {cap}")
        self.log_embed(win, f"Loaded video: {path} ({w}x{h}, frames={nframes}, fps={fps:.2f})")

    def clear_embedding(self, win):
        win.uploaded_frames = None
        win.video_meta = None
        win.stego_frames = None
        win.preview_label.configure(image='', text="No video uploaded")
        win.secret_txt.delete('1.0', tk.END)
        win.pwd_entry.delete(0, tk.END)
        win.rows_var.set("Rows: -")
        win.cols_var.set("Cols: -")
        win.frames_var.set("Frames: -")
        win.fps_var.set("FPS: -")
        win.cap_var.set("Capacity (bits): -")
        win.log_txt.delete('1.0', tk.END)

    def embed_action(self, win):
        if not getattr(win, 'uploaded_frames', None):
            messagebox.showerror("No video", "Please upload a cover video first.")
            return
        secret = win.secret_txt.get('1.0', tk.END).rstrip("\n")
        if not secret:
            messagebox.showerror("No secret", "Enter secret text to embed.")
            return
        password = win.pwd_entry.get()
        if not password:
            messagebox.showerror("Password", "Enter a password for encryption.")
            return
        try:
            bits_rep, raw = make_payload_bits(password, secret)
        except Exception as e:
            self.log_embed(win, f"Encryption failed: {e}")
            traceback.print_exc()
            messagebox.showerror("Encryption error", str(e))
            return
        frames = win.uploaded_frames
        h, w = frames[0].shape[:2]
        capacity_bits = len(frames) * h * w
        self.log_embed(win, f"Payload bits (with redundancy): {bits_rep.size}; Capacity bits: {capacity_bits}")
        if bits_rep.size > capacity_bits:
            messagebox.showerror("Too large", f"Payload ({bits_rep.size} bits) exceeds capacity ({capacity_bits} bits).")
            return
        try:
            stego_frames = embed_bits_into_frames(frames, bits_rep)
        except Exception as e:
            self.log_embed(win, f"Embedding error: {e}")
            traceback.print_exc()
            messagebox.showerror("Embedding error", str(e))
            return
        win.stego_frames = stego_frames
        win.raw_preview = raw[:64]
        self.log_embed(win, "Embedding done in memory.")
        messagebox.showinfo("Done", "Message embedded into video in memory. Save the stego video.")
        img_bgr = stego_frames[0]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb).resize((500, 320))
        imgtk = ImageTk.PhotoImage(img)
        win.preview_label.configure(image=imgtk, text="")
        win.preview_label.image = imgtk

    def save_stego_video(self, win):
        if not getattr(win, 'stego_frames', None):
            messagebox.showinfo("Info", "No stego video generated yet. Use Embed first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".mkv", filetypes=[("MKV (FFV1)", "*.mkv"), ("AVI", "*.avi")])
        if not save_path:
            return
        meta = win.video_meta
        try:
            write_video_frames_ffv1(save_path, win.stego_frames, meta['fps'], meta['width'], meta['height'])
        except Exception as e:
            self.log_embed(win, f"Save failed: {e}")
            traceback.print_exc()
            messagebox.showerror("Save failed", f"{e}")
            return
        self.log_embed(win, f"Saved stego to {save_path}")
        messagebox.showinfo("Saved", f"Stego video saved to:\n{save_path}")
        try:
            frames_saved, fps_s, w_s, h_s = read_video_frames(save_path)
            total_bits = len(frames_saved) * frames_saved[0].shape[0] * frames_saved[0].shape[1]
            scan_bits = min(total_bits, MAX_VERIFY_BYTES * 8 * BIT_REP)
            bits = extract_bits_from_frames(frames_saved, scan_bits)
            if bits.size % BIT_REP != 0:
                useful_bits = bits[:-(bits.size % BIT_REP)]
            else:
                useful_bits = bits
            groups = useful_bits.reshape((-1, BIT_REP))
            votes = (np.sum(groups, axis=1) > (BIT_REP//2)).astype(np.uint8)
            raw = np.packbits(votes).tobytes()
            if raw.find(SYNC * SYNC_REPEAT) == -1:
                self.log_embed(win, "Post-save verification: SYNC not found (codec may have altered bits).")
                messagebox.showwarning("Verification failed", "Saved file does not show the SYNC marker in a quick scan. Ensure you used MKV with FFV1 (lossless).")
            else:
                self.log_embed(win, "Post-save verification: SYNC found in saved file.")
                messagebox.showinfo("Verified", "Saved file appears to contain SYNC marker (good).")
        except Exception as e:
            self.log_embed(win, f"Verification error: {e}")
            traceback.print_exc()
            messagebox.showwarning("Verification error", f"Could not fully verify saved file: {e}")

    def open_extraction_window(self):
        win = tk.Toplevel(self.root)
        win.title("Extraction — Extract")
        win.geometry(f"{WINDOW_WIDTH + 50}x{WINDOW_HEIGHT}")
        win.resizable(False, False)
        win.configure(bg=LIGHT_CYAN)
        try:
            win.attributes('-alpha', 0.95)
        except Exception:
            pass

        menu = tk.Frame(win, bg=LIGHT_CYAN, width=200)
        menu.place(x=14, y=14, width=200, height=WINDOW_HEIGHT - 28)
        mlabel = tk.Label(menu, text="Menu", font=("Segoe UI Semibold", 12), fg=NEON, bg=LIGHT_CYAN)
        mlabel.pack(pady=(16,8))

        btn_load = tk.Button(menu, text="Load Stego Video", width=16, command=lambda: self.load_stego(win))
        btn_extract = tk.Button(menu, text="Extract", width=16, command=lambda: self.extract_action(win))
        btn_clear = tk.Button(menu, text="Clear All", width=16, command=lambda: self.clear_extraction(win))
        btn_exit = tk.Button(menu, text="Exit", width=16, command=win.destroy)
        for b in (btn_load, btn_extract, btn_clear, btn_exit):
            b.pack(pady=8)
            style_secondary_button(b)

        preview_frame = tk.Frame(win, bg="#E6FBFF", relief='sunken', bd=1)
        preview_frame.place(x=234, y=14, width=520, height=420)
        preview_label = tk.Label(preview_frame, text="No video loaded", bg="#E6FBFF")
        preview_label.pack(expand=True)

        right = tk.Frame(win, bg=LIGHT_CYAN)
        right.place(x=770, y=14, width=320, height=420)
        tk.Label(right, text="Extracted Secret Message", font=("Segoe UI", 10, "bold"), bg=LIGHT_CYAN).pack(anchor='nw', padx=10, pady=(10,4))
        extracted_txt = scrolledtext.ScrolledText(right, wrap=tk.WORD, height=8)
        extracted_txt.pack(fill='x', padx=10)

        tk.Label(right, text="Password (to decrypt)", bg=LIGHT_CYAN).pack(anchor='nw', padx=10, pady=(8,2))
        pwd_entry = tk.Entry(right, show="*")
        pwd_entry.pack(fill='x', padx=10)

        log_frame = tk.LabelFrame(right, text="Log", bg=LIGHT_CYAN)
        log_frame.pack(fill='both', padx=10, pady=(6,12), expand=True)
        log_txt = scrolledtext.ScrolledText(log_frame, height=6)
        log_txt.pack(fill='both', expand=True)

        win.preview_label = preview_label
        win.extracted_txt = extracted_txt
        win.pwd_entry = pwd_entry
        win.log_txt = log_txt
        win.stego_frames = None
        win.video_meta = None

    def load_stego(self, win):
        path = filedialog.askopenfilename(title="Select Stego Video", filetypes=[("Video Files", "*.mkv;*.avi;*.mp4;*.mov;*.mkv"), ("All files","*.*")])
        if not path:
            return
        try:
            frames, fps, width, height = read_video_frames(path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read video: {e}")
            return
        win.stego_frames = frames
        win.video_meta = dict(fps=fps, width=width, height=height, path=path)
        frame_bgr = frames[0]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((500, 320))
        imgtk = ImageTk.PhotoImage(img)
        win.preview_label.configure(image=imgtk, text="")
        win.preview_label.image = imgtk
        win.extracted_txt.delete('1.0', tk.END)
        self.log_extract(win, f"Loaded stego file: {path} ({width}x{height}, frames={len(frames)})")

    def log_extract(self, win, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        print(line, end='')
        try:
            win.log_txt.insert(tk.END, line)
            win.log_txt.see(tk.END)
        except Exception:
            pass

    def clear_extraction(self, win):
        win.stego_frames = None
        win.video_meta = None
        win.preview_label.configure(image='', text="No video loaded")
        win.extracted_txt.delete('1.0', tk.END)
        win.pwd_entry.delete(0, tk.END)
        win.log_txt.delete('1.0', tk.END)

    def extract_action(self, win):
        if not getattr(win, 'stego_frames', None):
            messagebox.showerror("No video", "Please load a stego video first.")
            return
        password = win.pwd_entry.get()
        if not password:
            messagebox.showerror("Password", "Enter the password used to encrypt.")
            return
        try:
            payload = find_payload_in_frames(win.stego_frames)
        except Exception as e:
            self.log_extract(win, f"Extraction scan failed: {e}")
            traceback.print_exc()
            messagebox.showerror("Extraction error", f"{e}\nLikely causes: wrong file, codec re-encoding, or embedding didn't run.")
            return
        try:
            plaintext = decrypt_message(password, payload)
        except Exception as e:
            self.log_extract(win, f"Decryption failed: {e}")
            traceback.print_exc()
            messagebox.showerror("Decryption error", f"Failed to decrypt: {e}")
            return
        win.extracted_txt.delete('1.0', tk.END)
        win.extracted_txt.insert(tk.END, plaintext)
        self.log_extract(win, "Extraction & decryption successful.")
        messagebox.showinfo("Success", "Secret extracted and decrypted.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStegoApp(root)
    root.mainloop()
