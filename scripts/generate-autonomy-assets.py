from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

INSTALLER_SIZE = (164, 314)
INSTALLER_SUPERSAMPLE = 6
INSTALLER_FRAME_COUNT = 125  # 125 frames at 40 ms = one exact five-second loop.
INSTALLER_STATIC_PREFIX_FRAMES = 24  # Closing frame + prefix = 25 frames / one second.
MIN_PARTICLE_COUNT = 10_000
MAX_PARTICLE_COUNT = 100_000
PARTICLE_LEVELS = 40
MARK_POSITION = (20, 50)
MARK_SIZE = 112


@dataclass(frozen=True, slots=True)
class MarkParticle:
    x: float
    y: float
    outward: float
    turns: float
    speed_variation: float
    aspect_x: float
    aspect_y: float
    warp: float
    radius: float
    seed: float
    opacity: int


def contain(image: Image.Image, size: tuple[int, int], padding: int = 0) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    inner = (size[0] - padding * 2, size[1] - padding * 2)
    value = image.copy()
    value.thumbnail(inner, Image.Resampling.LANCZOS)
    canvas.alpha_composite(value, ((size[0] - value.width) // 2, (size[1] - value.height) // 2))
    return canvas


def installer_background(size: tuple[int, int]) -> Image.Image:
    """Build the single-hue crimson installer canvas with restrained depth."""
    value = Image.new("RGB", size)
    pixels = value.load()
    for y in range(size[1]):
        vertical = y / max(size[1] - 1, 1)
        for x in range(size[0]):
            horizontal = x / max(size[0] - 1, 1)
            normalized = (x / size[0] * INSTALLER_SIZE[0], y / size[1] * INSTALLER_SIZE[1])
            glow = max(0.0, 1.0 - math.dist(normalized, (137, 54)) / 205)
            vignette = max(0.0, math.dist(normalized, (82, 145)) / 245)
            pixels[x, y] = (
                round(244 - 96 * vertical + 15 * glow - 10 * vignette),
                round(45 - 22 * vertical + 19 * glow - 4 * horizontal),
                round(82 - 31 * vertical + 16 * glow),
            )
    return value


def scatter_timeline(phase: float) -> float:
    """Expand and contract continuously throughout the four-second motion segment."""
    return 0.5 - 0.5 * math.cos(phase * math.tau)


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def motion_timeline(phase: float) -> float:
    """Map one static second and four moving seconds onto a seamless loop."""
    static_endpoint = (INSTALLER_STATIC_PREFIX_FRAMES - 1) / INSTALLER_FRAME_COUNT
    moving_span = (INSTALLER_FRAME_COUNT - INSTALLER_STATIC_PREFIX_FRAMES) / INSTALLER_FRAME_COUNT
    if phase <= static_endpoint:
        return 0.0
    return min(1.0, (phase - static_endpoint) / moving_span)


def visible_particle_count(motion: float) -> int:
    """Use the full reservoir near the bat and ten thousand particles at peak dispersion."""
    assembled = 1.0 - scatter_timeline(motion)
    density = smoothstep(assembled)
    return MIN_PARTICLE_COUNT + round((MAX_PARTICLE_COUNT - MIN_PARTICLE_COUNT) * density)


def build_mark_particles(mark: Image.Image) -> tuple[Image.Image, list[MarkParticle]]:
    """Sample the complete bat alpha mask so every region can dissolve into particles."""
    sprite = contain(white_silhouette(mark), (MARK_SIZE, MARK_SIZE), 4)
    alpha = sprite.getchannel("A")
    pixels = alpha.load()
    opaque = [
        (x + 0.5, y + 0.5) for y in range(MARK_SIZE) for x in range(MARK_SIZE) if pixels[x, y] >= 96
    ]
    edges = alpha.filter(ImageFilter.FIND_EDGES)
    edge_pixels = edges.load()
    outline = [
        (x + 0.5, y + 0.5)
        for y in range(1, MARK_SIZE - 1)
        for x in range(1, MARK_SIZE - 1)
        if pixels[x, y] >= 48 and edge_pixels[x, y] >= 34
    ]

    rng = random.Random(8052026)
    interior_count = round(MAX_PARTICLE_COUNT * 0.76)
    selected = rng.choices(opaque, k=interior_count)
    selected.extend(rng.choices(outline or opaque, k=MAX_PARTICLE_COUNT - interior_count))
    rng.shuffle(selected)

    center_x = MARK_SIZE / 2
    center_y = MARK_SIZE / 2
    particles: list[MarkParticle] = []

    def fraction(level: int) -> float:
        return level / (PARTICLE_LEVELS - 1)

    for index, (x, y) in enumerate(selected):
        x += rng.uniform(-0.48, 0.48)
        y += rng.uniform(-0.48, 0.48)
        radial_x = x - center_x
        radial_y = y - center_y
        length = max(1.0, math.hypot(radial_x, radial_y))

        opacity_level = index % PARTICLE_LEVELS
        size_level = (index * 11) % PARTICLE_LEVELS
        outward_level = (index * 17) % PARTICLE_LEVELS
        speed_level = (index * 23) % PARTICLE_LEVELS
        aspect_x_level = (index * 29) % PARTICLE_LEVELS
        aspect_y_level = (index * 31) % PARTICLE_LEVELS
        warp_level = (index * 37) % PARTICLE_LEVELS

        distance = (11.0 + 25.0 * fraction(outward_level)) * (0.72 + min(length / 55.0, 1.0) * 0.42)
        radius = 0.10 + 0.52 * fraction(size_level) ** 1.7
        particles.append(
            MarkParticle(
                x=x,
                y=y,
                outward=distance,
                turns=float(rng.choice((2, 2, 2, 3, 3, 4))),
                speed_variation=-0.72 + 1.44 * fraction(speed_level),
                aspect_x=0.72 + 0.63 * fraction(aspect_x_level),
                aspect_y=0.76 + 0.50 * fraction(aspect_y_level),
                warp=0.03 + 0.16 * fraction(warp_level),
                radius=radius,
                seed=rng.random(),
                opacity=72 + round(154 * fraction(opacity_level)),
            )
        )
    return sprite, particles


def draw_mark_disintegration(
    canvas: Image.Image,
    mark_sprite: Image.Image,
    particles: list[MarkParticle],
    phase: float,
    scale: int,
) -> None:
    """Dissolve the entire bat into particles, then reverse every path to rebuild it."""
    motion = motion_timeline(phase)
    scatter = scatter_timeline(motion)
    visible_count = visible_particle_count(motion)
    assembled = 1.0 - scatter
    solid_mix = smoothstep((assembled - 0.72) / 0.28)

    solid_mark = mark_sprite.resize(
        (MARK_SIZE * scale, MARK_SIZE * scale),
        Image.Resampling.LANCZOS,
    )
    solid_position = (MARK_POSITION[0] * scale, MARK_POSITION[1] * scale)
    if scatter <= 1e-9:
        canvas.alpha_composite(solid_mark, solid_position)
        return

    core = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core)
    glow = Image.new("RGB", canvas.size, (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)

    center_x = MARK_POSITION[0] + MARK_SIZE / 2
    center_y = MARK_POSITION[1] + MARK_SIZE / 2

    def spiral_position(particle: MarkParticle, current_phase: float) -> tuple[float, float]:
        base_x = particle.x - MARK_SIZE / 2
        base_y = particle.y - MARK_SIZE / 2
        base_radius = math.hypot(base_x, base_y)
        base_angle = math.atan2(base_y, base_x)
        current_scatter = scatter_timeline(current_phase)
        seed_angle = particle.seed * math.tau
        speed_wave = math.sin(current_phase * math.tau + seed_angle) - math.sin(seed_angle)
        angle = (
            base_angle
            + particle.turns * math.tau * current_phase
            + particle.speed_variation * speed_wave
        )
        radial_wave = 1.0 + particle.warp * current_scatter * (
            0.62 * math.sin(angle * 3.0 + seed_angle) + 0.38 * math.sin(angle * 5.0 - seed_angle)
        )
        radius = (base_radius + particle.outward * current_scatter) * radial_wave
        aspect_x = 1.0 + (particle.aspect_x - 1.0) * current_scatter
        aspect_y = 1.0 + (particle.aspect_y - 1.0) * current_scatter
        cross_warp = particle.warp * current_scatter * radius
        lift_x = 3.4 * current_scatter
        lift_y = -3.0 * current_scatter
        return (
            center_x
            + math.cos(angle) * radius * aspect_x
            + math.sin(angle * 2.0 + seed_angle) * cross_warp * 0.22
            + lift_x,
            center_y
            + math.sin(angle) * radius * aspect_y
            + math.cos(angle * 3.0 - seed_angle) * cross_warp * 0.18
            + lift_y,
        )

    for index in range(visible_count):
        particle = particles[index]
        x, y = spiral_position(particle, motion)
        x += math.sin((motion * particle.turns + particle.seed) * math.tau) * 0.34 * scatter
        y += math.cos((motion * particle.turns + particle.seed) * math.tau) * 0.34 * scatter
        radius_px = particle.radius * scale
        center = (x * scale, y * scale)
        trail_strength = math.sin(math.pi * motion)
        if (
            index < MIN_PARTICLE_COUNT
            and trail_strength > 0.03
            and (particle.radius > 0.34 or index % 29 == 0)
        ):
            prior_motion = max(0.0, motion - 0.008)
            prior_x, prior_y = spiral_position(particle, prior_motion)
            core_draw.line(
                (
                    prior_x * scale,
                    prior_y * scale,
                    center[0],
                    center[1],
                ),
                fill=(
                    255,
                    221,
                    232,
                    round(particle.opacity * 0.46 * trail_strength),
                ),
                width=max(1, round(radius_px * 0.72)),
            )

        if (
            index < MIN_PARTICLE_COUNT
            and scatter > 0.02
            and (particle.radius > 0.46 or index % 79 == 0)
        ):
            halo = (particle.radius * 3.1 + 0.8) * scale
            glow_value = round((14 + particle.opacity * 0.17) * scatter)
            glow_draw.ellipse(
                (
                    center[0] - halo,
                    center[1] - halo,
                    center[0] + halo,
                    center[1] + halo,
                ),
                fill=(glow_value, glow_value // 2, glow_value // 2),
            )
        particle_green = round(239 + 16 * solid_mix)
        particle_blue = round(244 + 11 * solid_mix)
        core_draw.ellipse(
            (
                center[0] - radius_px,
                center[1] - radius_px,
                center[0] + radius_px,
                center[1] + radius_px,
            ),
            fill=(255, particle_green, particle_blue, particle.opacity),
        )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=1.35 * scale))
    screened = ImageChops.screen(canvas.convert("RGB"), glow)
    canvas.paste(screened.convert("RGBA"))
    canvas.alpha_composite(core)
    if solid_mix > 0:
        solid_alpha = solid_mark.getchannel("A").point(lambda value: round(value * solid_mix))
        solid_mark.putalpha(solid_alpha)
        canvas.alpha_composite(solid_mark, solid_position)


def build_installer_frame(
    mark_sprite: Image.Image,
    particles: list[MarkParticle],
    bold: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    regular: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    phase: float,
    background: Image.Image,
    scale: int,
) -> Image.Image:
    sidebar = background.copy().convert("RGBA")
    draw_mark_disintegration(sidebar, mark_sprite, particles, phase, scale)
    draw = ImageDraw.Draw(sidebar)
    draw.text((20 * scale, 191 * scale), "AGENT", font=bold, fill=(255, 255, 255, 255))
    draw.line(
        (20 * scale, 219 * scale, 55 * scale, 219 * scale),
        fill=(255, 205, 218, 170),
        width=scale,
    )
    draw.text(
        (20 * scale, 227 * scale),
        "DroneDream",
        font=regular,
        fill=(255, 231, 236, 255),
    )
    return sidebar.convert("RGB").resize(INSTALLER_SIZE, Image.Resampling.LANCZOS)


def white_silhouette(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    output = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    output.putalpha(rgba.getchannel("A"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    mark = Image.open(args.source / "dronedream-agent-mark.png").convert("RGBA")
    lockup = Image.open(args.source / "dronedream-agent-lockup.png").convert("RGBA")

    public = args.repo / "app" / "frontend" / "public" / "brand"
    public.mkdir(parents=True, exist_ok=True)
    mark.save(public / "dronedream-agent-mark.png", optimize=True)
    lockup.save(public / "dronedream-agent-lockup.png", optimize=True)

    icons = args.repo / "app" / "desktop" / "src-tauri" / "icons"
    icons.mkdir(parents=True, exist_ok=True)
    icon_master = contain(mark, (512, 512), 28)
    for size, name in ((32, "32x32.png"), (128, "128x128.png"), (256, "128x128@2x.png")):
        icon_master.resize((size, size), Image.Resampling.LANCZOS).save(icons / name, optimize=True)
    icon_master.save(
        icons / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )

    installer = args.repo / "app" / "desktop" / "src-tauri" / "installer"
    installer.mkdir(parents=True, exist_ok=True)
    bold_path = Path("C:/Windows/Fonts/segoeuib.ttf")
    regular_path = Path("C:/Windows/Fonts/segoeui.ttf")
    bold = (
        ImageFont.truetype(str(bold_path), 18 * INSTALLER_SUPERSAMPLE)
        if bold_path.exists()
        else ImageFont.load_default()
    )
    regular = (
        ImageFont.truetype(str(regular_path), 10 * INSTALLER_SUPERSAMPLE)
        if regular_path.exists()
        else ImageFont.load_default()
    )
    render_size = tuple(value * INSTALLER_SUPERSAMPLE for value in INSTALLER_SIZE)
    background = installer_background(render_size)
    mark_sprite, particles = build_mark_particles(mark)
    build_installer_frame(
        mark_sprite,
        particles,
        bold,
        regular,
        0,
        background,
        INSTALLER_SUPERSAMPLE,
    ).save(installer / "sidebar.bmp")

    animation = installer / "animation"
    animation.mkdir(parents=True, exist_ok=True)
    for stale in animation.glob("sidebar-frame-*.bmp"):
        stale.unlink()
    for index in range(INSTALLER_FRAME_COUNT):
        phase = index / INSTALLER_FRAME_COUNT
        build_installer_frame(
            mark_sprite,
            particles,
            bold,
            regular,
            phase,
            background,
            INSTALLER_SUPERSAMPLE,
        ).save(animation / f"sidebar-frame-{index:02d}.bmp")


if __name__ == "__main__":
    main()
