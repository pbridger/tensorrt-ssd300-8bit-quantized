import sys, io
import time
import numpy as np
import threading

from PIL import Image
from pynvml import * # all functions have nvml prefix
nvmlInit()


dot = '•'
space = ' '
horizontal_line = '\u2500'
vertical_line = '\u2502'
fill = '\u2588'
tau = '\u03A4'
background_color = np.array([0x28, 0x2d, 0x35]) # svg-term background
foreground_color = np.array([128, 128, 128])
cold = np.array([0x3f, 0x5e, 0xfb])
hot = np.array([0xfc, 0x46, 0x6b])
not_nv = np.array([0x83, 0x60, 0xc3])
nv = np.array([0x2e, 0xbf, 0x91])
sky = np.array([0x1c, 0x92, 0xd2])
foreground_color = np.array([0xf2, 0xfc, 0xfe])
summer_low = np.array([0x22, 0xc1, 0xc3])
summer_hi = np.array([0xfd, 0xbb, 0x2d])


def write(s, stdout=sys.stdout):
    stdout.write(s)


def reset():
    write('\033c')


def wrap_color(rgb, s):
    r, g, b = map(int, rgb)
    return f'\033[38;2;{r};{g};{b}m{s}\033[0m'


def pixel_seq_to_ascii(pixel_rgb, char, count):
    if char == space or (pixel_rgb == background_color).all():
        return space * count
    return wrap_color(pixel_rgb, char * count)


def pixels_to_ascii(pixels, chars):
    rendered_rows = []
    for row in range(pixels.shape[0]):
        row_elements = []
        current_pixel, current_char = None, None
        pixel_count = 0

        for column in range(pixels.shape[1]):
            new_pixel, new_char = pixels[row, column], chars[row, column]
            if current_pixel is not None and new_char == current_char and (new_pixel == current_pixel).all():
                pixel_count += 1
            else:
                if current_pixel is not None:
                    row_elements.append(pixel_seq_to_ascii(current_pixel, current_char, pixel_count))
                current_pixel = new_pixel
                current_char = new_char
                pixel_count = 1

        if current_pixel is not None:
            row_elements.append(pixel_seq_to_ascii(current_pixel, current_char, pixel_count))
        rendered_rows.append(''.join(row_elements))

    return '\n'.join(rendered_rows)


def data_to_dot_matrix(xs, ys, size_wh, y_lim=(None, None)):
    dw = len(xs)
    y_min, y_max = y_lim
    if y_min is None:
        y_min = min(ys)
    if y_max is None:
        y_max = max(ys)
    dh = y_max - y_min

    w, h = size_wh
    dot_matrix = np.zeros((h, w), dtype=np.float32)
    values = np.zeros((w,), dtype=np.float32)

    for c in range(w):
        dc = dw * c / w
        dc, dc_fraction = int(dc // 1), dc % 1
        dy = ys[dc]
        if dc + 1 < dw:
            dy += dc_fraction * (ys[dc+1] - ys[dc])
        dy_proportion = (dy - y_min) / dh
        r = max(0, min(1, dy_proportion)) * h
        r, r_fraction = int(r // 1), r % 1

        dot_matrix[h - r:, c] = 1.0
        if h - r - 1 >= 0:
            dot_matrix[h - r - 1, c] = r_fraction
        values[c] = dy_proportion

    return dot_matrix, values


def dot_matrix_to_pixels(dot_matrix, values, low=foreground_color, high=foreground_color):
    pixels = np.ones((*dot_matrix.shape, 3)) * background_color
    chars = np.full(dot_matrix.shape, space, dtype='<U1')

    chars[dot_matrix > 0] = dot

    colors = np.expand_dims(values, -1) * (high - low) + low

    dot_matrix = np.expand_dims(dot_matrix, -1)
    pixels = ((1 - dot_matrix) * background_color) + (dot_matrix * colors)

    return pixels, chars


def render_axes(pixels, chars, title, x_ticks, y_ticks):
    ph, pw, pd = pixels.shape
    new_pixels = np.zeros((ph + 3, pw + 8, pd), dtype=pixels.dtype)
    new_pixels[1:ph + 1, :pw, :] = pixels
    new_chars = np.full((ph + 3, pw + 8), space, dtype='<U1')
    new_chars[1:ph + 1, :pw] = chars

    new_chars[ph + 1, :pw + 1] = horizontal_line
    new_pixels[ph + 1, :pw + 1] = foreground_color
    new_chars[1:ph + 1, pw + 1] = vertical_line
    new_pixels[1:ph + 1, pw + 1] = foreground_color

    # title
    x = (pw - len(title)) // 2
    new_chars[0, x:x + len(title)] = np.array(list(title))
    new_pixels[0, x:x + len(title)] = foreground_color

    # x-axis
    new_chars[ph + 2, pw + 1] = tau
    new_pixels[ph + 2, pw + 1] = foreground_color
    for x, label in x_ticks:
        new_chars[ph + 1, x] = '\u252C'
        new_chars[ph + 2, x:x + len(label)] = np.array(list(label))
        new_pixels[ph + 1, x] = foreground_color
        new_pixels[ph + 2, x:x + len(label)] = foreground_color

    # y-axis
    for y, label in y_ticks:
        label = ' ' + label
        y = max(0, y)
        new_chars[y + 1, pw + 1] = '\u251C'
        new_chars[y + 1, pw + 2:pw + 2 + len(label)] = np.array(list(label))
        new_pixels[y + 1, pw + 1] = foreground_color
        new_pixels[y + 1, pw + 2:pw + 2 + len(label)] = foreground_color

    new_chars[ph + 1, pw + 1] = '\u2518' #'\u253C'
    new_pixels[ph + 1, pw + 1] = foreground_color

    return new_pixels, new_chars


def log_to_pixels(stdout, chart_wh):
    pw, ph = chart_wh
    pixels = np.ones((ph + 3, pw + 8, 3)) * background_color
    chars = np.full((ph + 3, pw + 8), space, dtype='<U1')

    lines = stdout.getvalue().strip().split('\n')[-chart_wh[1]:]
    for from_bottom, line in enumerate(reversed(lines)):
        num_chars = min(pw, len(line))
        chars[ph - from_bottom, 0:0+num_chars] = np.array(list(line[:pw]))
        pixels[ph - from_bottom, 0:0+num_chars] = foreground_color

    return pixels, chars


def bg_plot(num_gpus, sample_hz):
    gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(min(num_gpus, nvmlDeviceGetCount()))]

    chart_wh = (50, 10)
    max_data_points = chart_wh[0]
    fps = [0 for i in range(max_data_points)]
    fps_lock = threading.Lock()

    def bg_thread_fn(chart_wh, max_data_points, fps, fps_lock):
        sample_period = 1 / sample_hz

        fps_min, fps_max = 0, 1500
        temp_min, temp_max = 40, 90
        gpu_min, gpu_max = 0, 100
        mem_min, mem_max = 0, 11
        temp = [0 for i in range(max_data_points)]
        gpu_util = [0 for i in range(max_data_points)]
        mem_used = [0 for i in range(max_data_points)]

        captured_stdout = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = captured_stdout
        captured_stderr = io.StringIO()
        real_stderr = sys.stderr
        # sys.stderr = captured_stderr
        try:
            while True:
                sample_time = time.time()

                with fps_lock:
                    if fps[-1] is None:
                        return
                    fps.append(fps[-1])

                for i, gpu in enumerate(gpu_handles):
                    if i > 0:
                        break

                    temp.append(nvmlDeviceGetTemperature(gpu, 0))

                    mem = nvmlDeviceGetMemoryInfo(gpu)
                    mem_used.append(mem.used / 2 ** 30)

                    util = nvmlDeviceGetUtilizationRates(gpu)
                    gpu_util.append(util.gpu)

                temp = temp[-max_data_points:]
                gpu_util = gpu_util[-max_data_points:]
                mem_used = mem_used[-max_data_points:]
                while len(fps) > max_data_points:
                    fps.pop(0)
                x = [t * sample_period for t in range(-len(temp), 0)]

                reset()

#                 log_pixels, log_chars = log_to_pixels(captured_stdout, chart_wh)

                fps_dm, fps_values = data_to_dot_matrix(x, fps, chart_wh, y_lim=(fps_min, fps_max))
                fps_pixels, fps_chars = dot_matrix_to_pixels(fps_dm, fps_values, low=sky, high=foreground_color)
                fps_pixels, fps_chars = render_axes(
                    fps_pixels, fps_chars,
                    'THROUGHPUT (FPS)',
                    [(0, f'{tau}{int(x[0])}s')],
                    [(0, f'{fps_max}'), (chart_wh[1], f'{fps_min}'), (int(chart_wh[1] * (fps_max - fps[-1]) / (fps_max - fps_min)), f'{fps[-1]:.0f}')]
                )

                temp_dm, temp_values = data_to_dot_matrix(x, temp, chart_wh, y_lim=(temp_min, temp_max))
                temp_pixels, temp_chars = dot_matrix_to_pixels(temp_dm, temp_values, low=cold, high=hot)
                temp_pixels, temp_chars = render_axes(
                    temp_pixels, temp_chars,
                    'GPU TEMP (C)',
                    [(0, f'{tau}{int(x[0])}s')],
                    [(0, f'{temp_max}C'), (chart_wh[1], f'{temp_min}C'), (int(chart_wh[1] * (temp_max - temp[-1]) / (temp_max - temp_min)), f'{temp[-1]:.0f}C')]
                )

                gpu_dm, gpu_values = data_to_dot_matrix(x, gpu_util, chart_wh, y_lim=(gpu_min, gpu_max))
                gpu_pixels, gpu_chars = dot_matrix_to_pixels(gpu_dm, gpu_values, low=not_nv, high=nv)
                gpu_pixels, gpu_chars = render_axes(
                    gpu_pixels, gpu_chars,
                    'GPU UTILIZATION (%)',
                    [(0, f'{tau}{int(x[0])}s')],
                    [(0, f'{gpu_max}%'), (chart_wh[1], f'{gpu_min:3.0f}%'), (int(chart_wh[1] * (gpu_max - gpu_util[-1]) / (gpu_max - gpu_min)), f'{gpu_util[-1]:3.0f}%')]
                )

                mem_dm, mem_values = data_to_dot_matrix(x, mem_used, chart_wh, y_lim=(mem_min, mem_max))
                mem_pixels, mem_chars = dot_matrix_to_pixels(mem_dm, mem_values, low=summer_low, high=summer_hi)
                mem_pixels, mem_chars = render_axes(
                    mem_pixels, mem_chars,
                    'GPU MEM (GB)',
                    [(0, f'{tau}{int(x[0])}s')],
                    [(0, f'{mem_max:3.1f}'), (chart_wh[1], f'{mem_min:3.1f}'), (int(chart_wh[1] * (mem_max - mem_used[-1]) / (mem_max - mem_min)), f'{mem_used[-1]:3.1f}')]
                )

                l_pixels = np.concatenate((
                    fps_pixels,
                    temp_pixels,
                ), axis=0)

                l_chars = np.concatenate((
                    fps_chars,
                    temp_chars,
                ), axis=0)

                r_pixels = np.concatenate((
                    gpu_pixels,
                    mem_pixels,
                ), axis=0)

                r_chars = np.concatenate((
                    gpu_chars,
                    mem_chars,
                ), axis=0)

                pixels = np.concatenate((l_pixels, r_pixels), axis=1)
                chars = np.concatenate((l_chars, r_chars), axis=1)

                time.sleep(0.001)
                write(pixels_to_ascii(pixels, chars) + '\n', stdout=real_stdout)

                time.sleep(max(0, sample_period - (time.time() - sample_time)))
        finally:
            sys.stdout = real_stdout
            sys.stderr = real_stderr

    bg_thread = threading.Thread(target=bg_thread_fn, args=(chart_wh, max_data_points, fps, fps_lock), daemon=True)
    
    def gen_update_fps(fps_lock, fps):
        def update_fps(latest_fps):
            with fps_lock:
                fps[-1] = latest_fps
        return update_fps
    
    return gen_update_fps(fps_lock, fps), bg_thread

