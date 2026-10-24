
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import math
import os
import sys

import numpy as np
import pygame
import numpy as np
from PIL import Image, ImageFilter

MODEL_PATH = "model.npz"

INPUT_SIZE = 784
HIDDEN_1 = 16
HIDDEN_2 = 16
OUTPUT_SIZE = 10

CANVAS_RESOLUTION = 280
BRUSH_SIZE = 20

FPS = 60

BG = (8, 9, 11)
PANEL = (17, 18, 21)
PANEL_INNER = (12, 13, 16)
BORDER = (42, 44, 49)
TEXT = (235, 237, 241)
MUTED = (133, 138, 148)
DIM = (69, 74, 84)
ACCENT = (74, 190, 255)
ACCENT_BRIGHT = (127, 218, 255)
ACTIVE_GREEN = (111, 236, 173)

VISIBLE_INPUT_NODES = 28

pygame.init()
pygame.display.set_caption("Neural Network Visualiser")

display_info = pygame.display.Info()
monitor_w = max(display_info.current_w, 1280)
monitor_h = max(display_info.current_h, 720)

initial_size = (
    min(int(monitor_w * 0.90), 3100),
    min(int(monitor_h * 0.88), 1350),
)

screen = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
clock = pygame.time.Clock()

fullscreen = False

FONT_TITLE = pygame.font.SysFont("Segoe UI", 30, bold=True)
FONT_HEADING = pygame.font.SysFont("Segoe UI", 22, bold=True)
FONT_BODY = pygame.font.SysFont("Segoe UI", 18)
FONT_SMALL = pygame.font.SysFont("Segoe UI", 15)
FONT_TINY = pygame.font.SysFont("Segoe UI", 13)
FONT_DIGIT = pygame.font.SysFont("Segoe UI", 100, bold=True)
FONT_PERCENT = pygame.font.SysFont("Segoe UI", 23, bold=True)

drawing_surface = pygame.Surface(
    (CANVAS_RESOLUTION, CANVAS_RESOLUTION)
)
drawing_surface.fill((0, 0, 0))

drawing = False
last_point = None

prediction = None
prediction_confidence = 0.0
probabilities = np.zeros(10, dtype=np.float64)

last_input = np.zeros(784, dtype=np.float64)
last_activations = []
last_forward_time = 0.0

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path!r}.\n"
            "Put model.npz in the same folder as visualiser.py."
        )

    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    weights = []
    biases = []

    # Format used by your existing model.
    if all(
        f"w{i}" in keys and f"b{i}" in keys
        for i in range(3)
    ):
        for i in range(3):
            weights.append(np.asarray(data[f"w{i}"], dtype=np.float64))
            biases.append(np.asarray(data[f"b{i}"], dtype=np.float64))

    elif "weights" in keys and "biases" in keys:
        weights = [
            np.asarray(w, dtype=np.float64)
            for w in data["weights"]
        ]
        biases = [
            np.asarray(b, dtype=np.float64)
            for b in data["biases"]
        ]
    else:
        raise ValueError(
            "Unrecognised model format.\n"
            f"Found keys: {sorted(keys)}"
        )

    data.close()

    if len(weights) != 3 or len(biases) != 3:
        raise ValueError("Expected exactly three weight and bias layers.")

    expected_shapes = [
        (784, 16),
        (16, 16),
        (16, 10),
    ]

    for i, expected in enumerate(expected_shapes):
        if weights[i].shape != expected:
            raise ValueError(
                f"Layer {i} has weight shape {weights[i].shape}; "
                f"expected {expected}."
            )

        if biases[i].shape != (expected[1],):
            raise ValueError(
                f"Layer {i} has bias shape {biases[i].shape}; "
                f"expected {(expected[1],)}."
            )

    print(f"Loaded {path}")
    print("Architecture: 784 -> 16 -> 16 -> 10")

    return weights, biases


weights, biases = load_model(MODEL_PATH)

def sigmoid(x):
    # Numerically stable sigmoid.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def forward_pass(inputs):
    activations = [inputs]
    current = inputs

    for w, b in zip(weights, biases):
        current = sigmoid(np.dot(current, w) + b)
        activations.append(current)

    return activations


def predict(inputs):
    global prediction
    global prediction_confidence
    global probabilities
    global last_activations
    global last_forward_time

    start = pygame.time.get_ticks()

    last_activations = forward_pass(inputs)
    outputs = last_activations[-1]

    prediction = int(np.argmax(outputs))

    total = float(np.sum(outputs))

    if total > 1e-12:
        probabilities = outputs / total
    else:
        probabilities = np.full(10, 0.1)

    prediction_confidence = float(probabilities[prediction])
    last_forward_time = pygame.time.get_ticks() - start


def preprocess_canvas():
    """
    Convert the Pygame drawing canvas into a centred 28x28
    grayscale input suitable for the MNIST neural network.

    Returns:
        A flattened NumPy array of 784 float values in [0, 1].
    """

    raw = pygame.surfarray.array3d(drawing_surface)
    gray = np.mean(raw, axis=2).T.astype(np.float32) / 255.0

    ys, xs = np.where(gray > 0.05)

    if len(xs) == 0:
        return np.zeros(784, dtype=np.float32)

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    digit = gray[y0:y1, x0:x1]
    h, w = digit.shape

    scale = 20.0 / max(h, w)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    digit_image = Image.fromarray(
        np.clip(digit * 255, 0, 255).astype(np.uint8)
    )

    digit_image = digit_image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )


    digit_image = digit_image.filter(
        ImageFilter.GaussianBlur(radius=0.25)
    )

    resized = np.asarray(
        digit_image,
        dtype=np.float32,
    ) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    total = canvas.sum()

    if total > 1e-8:
        yy, xx = np.indices(canvas.shape)

        cx = float((xx * canvas).sum() / total)
        cy = float((yy * canvas).sum() / total)
   
        target_x = 13.5
        target_y = 13.5

        shift_x = int(round(target_x - cx))
        shift_y = int(round(target_y - cy))

        shifted = np.zeros_like(canvas)

        src_x0 = max(0, -shift_x)
        src_x1 = min(28, 28 - shift_x)
        src_y0 = max(0, -shift_y)
        src_y1 = min(28, 28 - shift_y)

        dst_x0 = max(0, shift_x)
        dst_x1 = min(28, 28 + shift_x)
        dst_y0 = max(0, shift_y)
        dst_y1 = min(28, 28 + shift_y)

        if src_x1 > src_x0 and src_y1 > src_y0:
            shifted[
                dst_y0:dst_y1,
                dst_x0:dst_x1,
            ] = canvas[
                src_y0:src_y1,
                src_x0:src_x1,
            ]

        canvas = shifted

    return canvas.flatten().astype(np.float32)


def update_prediction():
    global last_input

    last_input = preprocess_canvas()
    predict(last_input)


def clear_canvas():
    global prediction
    global prediction_confidence
    global probabilities
    global last_activations
    global last_input

    drawing_surface.fill((0, 0, 0))

    prediction = None
    prediction_confidence = 0.0
    probabilities = np.zeros(10)
    last_input = np.zeros(784)
    last_activations = []

def draw_text(surface, text, font, color, position):
    image = font.render(str(text), True, color)
    surface.blit(image, position)
    return image.get_rect(topleft=position)


def centered_text(surface, text, font, color, center):
    image = font.render(str(text), True, color)
    rect = image.get_rect(center=center)
    surface.blit(image, rect)
    return rect


def rounded_panel(surface, rect, color=PANEL, border=BORDER, radius=16):
    pygame.draw.rect(
        surface, color, rect, border_radius=radius
    )
    pygame.draw.rect(
        surface, border, rect, width=1, border_radius=radius
    )


def lerp_color(a, b, amount):
    amount = max(0.0, min(1.0, amount))
    return tuple(
        int(a[i] + (b[i] - a[i]) * amount)
        for i in range(3)
    )


def draw_glow(surface, position, radius, color, strength):
    """Draw a restrained, layered glow around active neurons."""

    strength = max(0.0, min(1.0, strength))

    if strength < 0.12:
        return

    x, y = int(position[0]), int(position[1])

    glow_color = lerp_color(
        (8, 9, 11), color, 0.35 + strength * 0.4
    )

    for extra, factor in ((9, 0.10), (6, 0.14), (3, 0.20)):
        r = radius + extra
        alpha_color = lerp_color(
            BG, glow_color, strength * factor * 2.5
        )
        pygame.draw.circle(surface, alpha_color, (x, y), r)

    pygame.draw.circle(
        surface,
        lerp_color(DIM, color, strength),
        (x, y),
        radius + 1,
    )


def draw_neuron(surface, position, radius, activation, outline_color):
    activation = float(np.clip(activation, 0.0, 1.0))

    fill = lerp_color(
        (23, 25, 30),
        ACCENT,
        activation * 0.80,
    )

    border = lerp_color(
        (63, 67, 76),
        outline_color,
        activation,
    )

    draw_glow(
        surface,
        position,
        radius,
        outline_color,
        activation,
    )

    pygame.draw.circle(
        surface,
        fill,
        (int(position[0]), int(position[1])),
        radius,
    )

    pygame.draw.circle(
        surface,
        border,
        (int(position[0]), int(position[1])),
        radius,
        width=1,
    )


def fit_font(base_size, minimum=12, maximum=34):
    size = max(minimum, min(maximum, base_size))
    return pygame.font.SysFont("Segoe UI", size)


def calculate_layout():
    width, height = screen.get_size()

    margin = max(18, min(36, width // 100))
    gap = max(14, min(24, width // 140))

    usable_width = width - 2 * margin - 2 * gap

    # Cap content width so ultrawide layouts remain balanced.
    usable_width = min(usable_width, 3200)

    left_w = int(usable_width * 0.235)
    right_w = int(usable_width * 0.235)
    network_w = usable_width - left_w - right_w

    content_x = (width - usable_width) // 2

    left = pygame.Rect(
        content_x,
        112,
        left_w,
        height - 154,
    )

    network = pygame.Rect(
        left.right + gap,
        112,
        network_w,
        height - 154,
    )

    right = pygame.Rect(
        network.right + gap,
        112,
        right_w,
        height - 154,
    )

    return left, network, right, margin, gap


def draw_header(surface, width):
    draw_text(
        surface,
        "NEURAL NETWORK",
        FONT_TITLE,
        TEXT,
        (32, 24),
    )

    draw_text(
        surface,
        "MNIST DIGIT CLASSIFIER",
        FONT_SMALL,
        MUTED,
        (34, 62),
    )

    status = "MODEL LOADED"
    status_surface = FONT_SMALL.render(status, True, ACTIVE_GREEN)

    status_rect = status_surface.get_rect(
        topright=(width - 32, 38)
    )

    # Keep the status label aligned on the right.
    surface.blit(status_surface, status_rect)

    pygame.draw.line(
        surface,
        BORDER,
        (32, 88),
        (width - 32, 88),
        1,
    )


def draw_input_preview(surface, rect):
    """Show the exact 28x28 input currently passed to the model."""
    rounded_panel(surface, rect, PANEL_INNER, BORDER, 10)

    draw_text(
        surface,
        "28 × 28 INPUT",
        FONT_SMALL,
        TEXT,
        (rect.x + 12, rect.y + 10),
    )

    draw_text(
        surface,
        "Model input",
        FONT_TINY,
        MUTED,
        (rect.x + 12, rect.y + 31),
    )

    # Nearest-neighbour scaling keeps individual pixels crisp.
    pixel_surface = pygame.Surface((28, 28))

    for y in range(28):
        for x in range(28):
            value = float(last_input[y * 28 + x])
            intensity = int(np.clip(value, 0.0, 1.0) * 255)
            pixel_surface.set_at(
                (x, y),
                (intensity, intensity, intensity),
            )

    preview_size = min(rect.width - 24, rect.height - 65)
    preview_size = max(28, preview_size)

    scaled = pygame.transform.scale(
        pixel_surface,
        (preview_size, preview_size),
    )

    preview_rect = pygame.Rect(
        rect.centerx - preview_size // 2,
        rect.y + 55,
        preview_size,
        preview_size,
    )

    surface.blit(scaled, preview_rect.topleft)

    pygame.draw.rect(
        surface,
        BORDER,
        preview_rect,
        width=1,
    )


def draw_canvas_panel(surface, rect):
    rounded_panel(surface, rect)

    pad = 24
    button_h = 40
    button_gap = 10
    bottom_pad = 24

    # Header
    draw_text(
        surface,
        "Drawing",
        FONT_HEADING,
        TEXT,
        (rect.x + pad, rect.y + pad),
    )

    draw_text(
        surface,
        "Draw a digit from 0 to 9",
        FONT_SMALL,
        MUTED,
        (rect.x + pad, rect.y + pad + 32),
    )

    # Bottom controls
    button_y = rect.bottom - bottom_pad - button_h
    button_w = min(220, rect.width - 2 * pad)

    clear_rect = pygame.Rect(
        rect.centerx - button_w // 2,
        button_y,
        button_w,
        button_h,
    )

    predict_rect = pygame.Rect(
        rect.centerx - button_w // 2,
        button_y - button_h - button_gap,
        button_w,
        button_h,
    )

    # Preview centred directly above the buttons
    preview_gap = 14
    preview_max_size = min(140, rect.width - 2 * pad)

    preview_bottom = predict_rect.y - preview_gap
    preview_top_limit = rect.y + 100

    preview_size = min(
        preview_max_size,
        max(1, preview_bottom - preview_top_limit - 12),
    )

    preview_rect = pygame.Rect(
        rect.centerx - preview_size // 2,
        preview_bottom - preview_size,
        preview_size,
        preview_size,
    )

    draw_input_preview(surface, preview_rect)

    # Drawing canvas occupies the space above the preview
    canvas_top = rect.y + 100
    canvas_bottom = preview_rect.y - 18

    canvas_size = max(
        1,
        min(
            rect.width - 2 * pad,
            canvas_bottom - canvas_top,
            680,
        ),
    )

    canvas_rect = pygame.Rect(
        rect.centerx - canvas_size // 2,
        0,
        canvas_size,
        canvas_size,
    )

    canvas_rect.centery = (canvas_top + canvas_bottom) // 2

    pygame.draw.rect(
        surface,
        (5, 6, 8),
        canvas_rect,
        border_radius=10,
    )

    scaled_canvas = pygame.transform.scale(
        drawing_surface,
        canvas_rect.size,
    )

    surface.blit(scaled_canvas, canvas_rect.topleft)

    pygame.draw.rect(
        surface,
        (48, 51, 59),
        canvas_rect,
        width=1,
        border_radius=10,
    )

    # Predict button
    rounded_panel(
        surface,
        predict_rect,
        PANEL_INNER,
        BORDER,
        9,
    )

    centered_text(
        surface,
        "PREDICT",
        FONT_SMALL,
        ACCENT_BRIGHT,
        predict_rect.center,
    )

    # Clear button
    rounded_panel(
        surface,
        clear_rect,
        PANEL_INNER,
        BORDER,
        9,
    )

    centered_text(
        surface,
        "CLEAR  [C]",
        FONT_SMALL,
        TEXT,
        clear_rect.center,
    )

    return canvas_rect, clear_rect, predict_rect


def evenly_spaced_positions(x, top, bottom, count):
    if count == 1:
        return [(x, (top + bottom) / 2)]

    return [
        (x, top + i * (bottom - top) / (count - 1))
        for i in range(count)
    ]


def draw_network_panel(surface, rect):
    rounded_panel(surface, rect)

    pad = 28

    draw_text(
        surface,
        "Network activity",
        FONT_HEADING,
        TEXT,
        (rect.x + pad, rect.y + 22),
    )

    draw_text(
        surface,
        "LIVE ACTIVATIONS",
        FONT_TINY,
        MUTED,
        (rect.x + pad, rect.y + 53),
    )

    draw_text(
        surface,
        "Brighter nodes and lines indicate stronger activity",
        FONT_SMALL,
        MUTED,
        (rect.x + pad, rect.y + 77),
    )

    # Graph dimensions adapt to both ultrawide and windowed layouts.
    graph_left = rect.x + 48
    graph_right = rect.right - 48
    graph_top = rect.y + 145
    graph_bottom = rect.bottom - 112

    graph_height = max(100, graph_bottom - graph_top)

    layer_xs = [
        graph_left,
        graph_left + (graph_right - graph_left) * 0.31,
        graph_left + (graph_right - graph_left) * 0.66,
        graph_right,
    ]

    # Sample 28 positions across the input image. This is purely
    # a visual summary; every input pixel still enters the model.
    sample_indices = np.linspace(
        0,
        783,
        VISIBLE_INPUT_NODES,
        dtype=int,
    )

    input_activations = last_input[sample_indices]

    if len(last_activations) == 4:
        hidden_1 = last_activations[1]
        hidden_2 = last_activations[2]
        output_activations = last_activations[3]
    else:
        hidden_1 = np.zeros(16)
        hidden_2 = np.zeros(16)
        output_activations = np.zeros(10)

    layer_values = [
        input_activations,
        hidden_1,
        hidden_2,
        output_activations,
    ]

    layer_counts = [
        VISIBLE_INPUT_NODES,
        16,
        16,
        10,
    ]

    layer_labels = [
        "INPUT",
        "HIDDEN 1",
        "HIDDEN 2",
        "OUTPUT",
    ]

    positions = []

    for layer_index, count in enumerate(layer_counts):
        # Leave a small margin above and below each layer.
        vertical_padding = max(12, graph_height * 0.025)

        top = graph_top + vertical_padding
        bottom = graph_bottom - vertical_padding

        # A denser input layer naturally appears taller.
        layer_positions = evenly_spaced_positions(
            layer_xs[layer_index],
            top,
            bottom,
            count,
        )

        positions.append(layer_positions)

    # Connections: each visible neuron links to every neuron
    # in the next layer. Input weights use the corresponding
    # sampled pixel indices from the original 784-pixel layer.
    for layer_index in range(3):
        source_positions = positions[layer_index]
        target_positions = positions[layer_index + 1]

        layer_weights = weights[layer_index]

        for source_index, source_pos in enumerate(source_positions):
            if layer_index == 0:
                actual_source_index = sample_indices[source_index]
            else:
                actual_source_index = source_index

            source_activation = float(
                layer_values[layer_index][source_index]
            )

            for target_index, target_pos in enumerate(target_positions):
                weight = float(
                    layer_weights[actual_source_index, target_index]
                )

                target_activation = float(
                    layer_values[layer_index + 1][target_index]
                )

                strength = min(abs(weight) / 2.0, 1.0)

                # Brightness depends on the weight and the
                # activity of both endpoints.
                activity = (
                    0.15
                    + 0.45 * source_activation
                    + 0.40 * target_activation
                )

                brightness = strength * activity

                if weight >= 0:
                    base = (27, 35, 44)
                    lit = (70, 174, 225)
                else:
                    base = (38, 29, 35)
                    lit = (214, 106, 125)

                line_color = lerp_color(
                    base,
                    lit,
                    brightness,
                )

                line_width = (
                    2 if brightness > 0.72 else 1
                )

                pygame.draw.line(
                    surface,
                    line_color,
                    (
                        int(source_pos[0]),
                        int(source_pos[1]),
                    ),
                    (
                        int(target_pos[0]),
                        int(target_pos[1]),
                    ),
                    line_width,
                )

    # Draw the neurons after the connections so they sit above
    # the lines and their activation glows remain visible.
    for layer_index, layer_positions in enumerate(positions):
        for neuron_index, position in enumerate(layer_positions):
            activation = float(
                layer_values[layer_index][neuron_index]
            )

            if layer_index == 0:
                radius = 5
            elif layer_index in (1, 2):
                radius = 9
            else:
                radius = 11

            outline_color = (
                ACTIVE_GREEN
                if layer_index == 3
                and prediction == neuron_index
                else ACCENT_BRIGHT
            )

            draw_neuron(
                surface,
                position,
                radius,
                activation,
                outline_color,
            )

    # Layer labels
    label_y = rect.bottom - 78

    for i, label in enumerate(layer_labels):
        centered_text(
            surface,
            label,
            FONT_SMALL,
            TEXT,
            (int(layer_xs[i]), label_y),
        )

    count_labels = [
        f"{INPUT_SIZE} pixels",
        "16 neurons",
        "16 neurons",
        "10 classes",
    ]

    for i, label in enumerate(count_labels):
        centered_text(
            surface,
            label,
            FONT_TINY,
            MUTED,
            (int(layer_xs[i]), label_y + 22),
        )


def draw_prediction_panel(surface, rect):
    rounded_panel(surface, rect)

    pad = 24

    draw_text(
        surface,
        "Prediction",
        FONT_HEADING,
        TEXT,
        (rect.x + pad, rect.y + 22),
    )

    draw_text(
        surface,
        "CLASSIFICATION RESULT",
        FONT_TINY,
        MUTED,
        (rect.x + pad, rect.y + 53),
    )

    result_top = rect.y + 88
    result_height = min(260, rect.height * 0.31)

    result_rect = pygame.Rect(
        rect.x + pad,
        result_top,
        rect.width - 2 * pad,
        result_height,
    )

    rounded_panel(
        surface,
        result_rect,
        PANEL_INNER,
        BORDER,
        12,
    )

    if prediction is None:
        centered_text(
            surface,
            "—",
            FONT_DIGIT,
            DIM,
            (
                result_rect.centerx,
                result_rect.y + result_rect.height * 0.43,
            ),
        )

        centered_text(
            surface,
            "Draw a digit to begin",
            FONT_SMALL,
            MUTED,
            (
                result_rect.centerx,
                result_rect.bottom - 32,
            ),
        )
    else:
        centered_text(
            surface,
            str(prediction),
            FONT_DIGIT,
            TEXT,
            (
                result_rect.centerx,
                result_rect.y + result_rect.height * 0.43,
            ),
        )

        percent = f"{prediction_confidence * 100:.1f}% relative score"

        centered_text(
            surface,
            percent,
            FONT_SMALL,
            ACTIVE_GREEN,
            (
                result_rect.centerx,
                result_rect.bottom - 32,
            ),
        )

    # Output ranking
    bars_top = result_rect.bottom + 28

    draw_text(
        surface,
        "Output scores",
        FONT_BODY,
        TEXT,
        (rect.x + pad, bars_top),
    )

    draw_text(
        surface,
        "0–9",
        FONT_TINY,
        MUTED,
        (rect.right - pad - 25, bars_top + 3),
    )

    bar_start_y = bars_top + 35
    bar_gap = max(27, min(38, (rect.bottom - bar_start_y - 90) // 10))

    label_font = fit_font(int(bar_gap * 0.47), 12, 18)

    max_bar_w = rect.width - 2 * pad - 100
    max_bar_w = max(40, max_bar_w)

    for digit in range(10):
        y = bar_start_y + digit * bar_gap

        if y + bar_gap > rect.bottom - 60:
            break

        value = float(probabilities[digit])
        is_best = prediction == digit

        draw_text(
            surface,
            str(digit),
            label_font,
            TEXT if is_best else MUTED,
            (rect.x + pad, y - 2),
        )

        bar_x = rect.x + pad + 30
        bar_y = y + 5

        background_rect = pygame.Rect(
            bar_x,
            bar_y,
            max_bar_w,
            7,
        )

        pygame.draw.rect(
            surface,
            (37, 39, 45),
            background_rect,
            border_radius=4,
        )

        fill_w = int(max_bar_w * max(0.0, min(1.0, value)))

        if fill_w > 0:
            fill_rect = pygame.Rect(
                bar_x,
                bar_y,
                fill_w,
                7,
            )

            bar_color = (
                ACTIVE_GREEN if is_best else ACCENT
            )

            pygame.draw.rect(
                surface,
                bar_color,
                fill_rect,
                border_radius=4,
            )

        percentage_text = f"{value * 100:.1f}%"

        draw_text(
            surface,
            percentage_text,
            FONT_TINY,
            ACTIVE_GREEN if is_best else MUTED,
            (bar_x + max_bar_w + 9, y),
        )

    footer_y = rect.bottom - 45

    draw_text(
        surface,
        "C  Clear",
        FONT_TINY,
        MUTED,
        (rect.x + pad, footer_y),
    )

    draw_text(
        surface,
        "F11  Fullscreen",
        FONT_TINY,
        MUTED,
        (rect.x + pad + 95, footer_y),
    )


def draw_interface():
    screen.fill(BG)

    width, height = screen.get_size()

    draw_header(screen, width)

    left, network, right, margin, gap = calculate_layout()

    canvas_rect, clear_rect, predict_rect = draw_canvas_panel(
        screen, left
    )

    draw_network_panel(screen, network)
    draw_prediction_panel(screen, right)

    # Footer separator
    pygame.draw.line(
        screen,
        BORDER,
        (32, height - 27),
        (width - 32, height - 27),
        1,
    )

    draw_text(
        screen,
        "784 → 16 → 16 → 10",
        FONT_TINY,
        MUTED,
        (32, height - 20),
    )

    draw_text(
        screen,
        "NumPy  •  Sigmoid  •  MNIST",
        FONT_TINY,
        MUTED,
        (220, height - 20),
    )

    return canvas_rect, clear_rect, predict_rect


def toggle_fullscreen():
    global screen, fullscreen

    fullscreen = not fullscreen

    if fullscreen:
        screen = pygame.display.set_mode(
            (0, 0),
            pygame.FULLSCREEN,
        )
    else:
        screen = pygame.display.set_mode(
            initial_size,
            pygame.RESIZABLE,
        )


running = True
canvas_rect = pygame.Rect(0, 0, 1, 1)
clear_rect = pygame.Rect(0, 0, 1, 1)
predict_rect = pygame.Rect(0, 0, 1, 1)

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.VIDEORESIZE and not fullscreen:
            screen = pygame.display.set_mode(
                (max(900, event.w), max(600, event.h)),
                pygame.RESIZABLE,
            )

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if fullscreen:
                    toggle_fullscreen()
                else:
                    running = False

            elif event.key == pygame.K_F11:
                toggle_fullscreen()

            elif event.key == pygame.K_c:
                clear_canvas()

            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                update_prediction()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse = event.pos

                if canvas_rect.collidepoint(mouse):
                    drawing = True

                    local_x = (
                        (mouse[0] - canvas_rect.x)
                        / canvas_rect.width
                        * CANVAS_RESOLUTION
                    )

                    local_y = (
                        (mouse[1] - canvas_rect.y)
                        / canvas_rect.height
                        * CANVAS_RESOLUTION
                    )

                    last_point = (int(local_x), int(local_y))

                    pygame.draw.circle(
                        drawing_surface,
                        (255, 255, 255),
                        last_point,
                        BRUSH_SIZE // 2,
                    )

                    update_prediction()

                elif clear_rect.collidepoint(mouse):
                    clear_canvas()

                elif predict_rect.collidepoint(mouse):
                    update_prediction()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                last_point = None

        elif event.type == pygame.MOUSEMOTION and drawing:
            if canvas_rect.collidepoint(event.pos):
                local_x = (
                    (event.pos[0] - canvas_rect.x)
                    / canvas_rect.width
                    * CANVAS_RESOLUTION
                )

                local_y = (
                    (event.pos[1] - canvas_rect.y)
                    / canvas_rect.height
                    * CANVAS_RESOLUTION
                )

                point = (int(local_x), int(local_y))

                if last_point is not None:
                    pygame.draw.line(
                        drawing_surface,
                        (255, 255, 255),
                        last_point,
                        point,
                        BRUSH_SIZE,
                    )

                pygame.draw.circle(
                    drawing_surface,
                    (255, 255, 255),
                    point,
                    BRUSH_SIZE // 2,
                )

                last_point = point
                update_prediction()

    canvas_rect, clear_rect, predict_rect = draw_interface()
    pygame.display.flip()

pygame.quit()
sys.exit()
