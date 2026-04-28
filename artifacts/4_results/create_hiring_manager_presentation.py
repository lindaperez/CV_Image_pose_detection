#!/usr/bin/env python3
"""Create a PowerPoint presentation for the exercise counting project.

This generator uses only the Python standard library and writes a valid PPTX
Open XML package. Figures are embedded from this results folder as SVG media.
"""

from __future__ import annotations

import html
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent
PPTX_PATH = OUT_DIR / "exercise_counting_hiring_manager_presentation.pptx"

EMU_PER_INCH = 914400
SLIDE_W = 13.333333
SLIDE_H = 7.5
SLIDE_CX = int(SLIDE_W * EMU_PER_INCH)
SLIDE_CY = int(SLIDE_H * EMU_PER_INCH)

COLORS = {
    "ink": "17212B",
    "muted": "52606D",
    "line": "D9E2EC",
    "soft": "F8FAFC",
    "soft_blue": "EAF3FF",
    "soft_green": "EAF8EE",
    "soft_rose": "FFF1F3",
    "blue": "4477AA",
    "green": "228833",
    "rose": "CC6677",
    "orange": "D97706",
    "white": "FFFFFF",
}


def emu(inches: float) -> int:
    return int(inches * EMU_PER_INCH)


def xml_text(value: str) -> str:
    return escape(value, {"'": "&apos;", '"': "&quot;"})


def attr(value: str) -> str:
    return html.escape(value, quote=True)


class Ids:
    def __init__(self) -> None:
        self.value = 1

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


def run(text: str, size: int = 22, color: str = COLORS["ink"], bold: bool = False) -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{size * 100}"{b}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Aptos"/></a:rPr>'
        f"<a:t>{xml_text(text)}</a:t></a:r>"
    )


def paragraph(text: str, size: int = 22, color: str = COLORS["ink"], bold: bool = False, bullet: bool = False) -> str:
    if bullet:
        ppr = '<a:pPr marL="285750" indent="-171450"><a:buChar char="•"/></a:pPr>'
    else:
        ppr = "<a:pPr/>"
    return f"<a:p>{ppr}{run(text, size=size, color=color, bold=bold)}<a:endParaRPr lang=\"en-US\" sz=\"{size * 100}\"/></a:p>"


def text_box(
    ids: Ids,
    x: float,
    y: float,
    w: float,
    h: float,
    paragraphs: list[str],
    name: str,
    fill: str | None = None,
    line: str | None = None,
    radius: str = "roundRect",
) -> str:
    shape_id = ids.next()
    fill_xml = "<a:noFill/>" if fill is None else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
    line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
    return f"""
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="{shape_id}" name="{attr(name)}"/>
          <p:cNvSpPr txBox="1"/>
          <p:nvPr/>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
          {fill_xml}
          {line_xml}
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" lIns="91440" tIns="68580" rIns="91440" bIns="68580"/>
          <a:lstStyle/>
          {''.join(paragraphs)}
        </p:txBody>
      </p:sp>
    """


def rect_shape(
    ids: Ids,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    name: str,
    line: str | None = None,
    radius: str = "roundRect",
) -> str:
    shape_id = ids.next()
    line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="{attr(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
          <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
          {line_xml}
        </p:spPr>
      </p:sp>
    """


def line_shape(ids: Ids, x1: float, y1: float, x2: float, y2: float, color: str = COLORS["line"], width: int = 19050) -> str:
    shape_id = ids.next()
    return f"""
      <p:cxnSp>
        <p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="Connector {shape_id}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/><a:ext cx="{emu(abs(x2 - x1))}" cy="{emu(abs(y2 - y1))}"/></a:xfrm>
          <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
          <a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
        </p:spPr>
      </p:cxnSp>
    """


def picture(ids: Ids, x: float, y: float, w: float, h: float, rel_id: str, name: str) -> str:
    pic_id = ids.next()
    return f"""
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="{pic_id}" name="{attr(name)}"/>
          <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="{rel_id}"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    """


def title(ids: Ids, text: str, subtitle: str | None = None) -> str:
    body = text_box(ids, 0.55, 0.28, 11.7, 0.5, [paragraph(text, 27, bold=True)], "Slide Title")
    if subtitle:
        body += text_box(ids, 0.57, 0.82, 11.6, 0.35, [paragraph(subtitle, 12, COLORS["muted"])], "Slide Subtitle")
    body += rect_shape(ids, 0.55, 1.17, 12.2, 0.02, COLORS["line"], "Divider", radius="rect")
    return body


def footer(ids: Ids, speaker: str, visual: str | None = None) -> str:
    left = f"Speaker: {speaker}"
    right = f"Visualization: {visual}" if visual else "Exercise Repetition Counting from Video"
    return (
        text_box(ids, 0.55, 7.05, 5.0, 0.25, [paragraph(left, 9, COLORS["muted"])], "Footer Left")
        + text_box(ids, 7.05, 7.05, 5.75, 0.25, [paragraph(right, 9, COLORS["muted"])], "Footer Right")
    )


def slide_xml(shapes: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{COLORS['white']}"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {shapes}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def rels_xml(relationships: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(
        f'  <Relationship Id="{rid}" Type="{rtype}" Target="{attr(target)}"/>'
        for rid, rtype, target in relationships
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rows}
</Relationships>
"""


def slide_rels(images: list[str]) -> str:
    relationships = [
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")
    ]
    for idx, image in enumerate(images, start=2):
        relationships.append((f"rId{idx}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{image}"))
    return rels_xml(relationships)


def card(ids: Ids, x: float, y: float, w: float, h: float, heading: str, body: str, fill: str) -> str:
    return text_box(
        ids,
        x,
        y,
        w,
        h,
        [paragraph(heading, 15, bold=True), paragraph(body, 11, COLORS["muted"])],
        heading,
        fill=fill,
        line=COLORS["line"],
    )


def metric_card(ids: Ids, x: float, y: float, label: str, value: str, detail: str, color: str) -> str:
    return text_box(
        ids,
        x,
        y,
        3.7,
        1.05,
        [paragraph(label, 12, COLORS["muted"], bold=True), paragraph(value, 24, color, bold=True), paragraph(detail, 10, COLORS["muted"])],
        label,
        fill=COLORS["soft"],
        line=color,
    )


def build_slides() -> list[dict[str, object]]:
    slides: list[dict[str, object]] = []

    ids = Ids()
    shapes = (
        rect_shape(ids, 0, 0, 13.333333, 7.5, COLORS["soft_blue"], "Background", radius="rect")
        + text_box(ids, 0.82, 1.0, 10.7, 1.05, [paragraph("Exercise Repetition Counting From Video", 34, bold=True)], "Main Title")
        + text_box(ids, 0.88, 2.06, 9.95, 0.55, [paragraph("Pose, RGB, temporal modeling, and exercise-dependent routing", 20, COLORS["muted"])], "Subtitle")
        + card(ids, 0.9, 3.05, 3.6, 1.22, "Problem", "Count repetitions directly from workout videos.", COLORS["white"])
        + card(ids, 4.85, 3.05, 3.6, 1.22, "Approach", "Compare pose, RGB, FSM, TCN, Transformer, and fusion branches.", COLORS["white"])
        + card(ids, 8.8, 3.05, 3.6, 1.22, "Outcome", "Routed architecture plus a squat runtime prototype.", COLORS["white"])
        + text_box(ids, 0.9, 6.58, 8.7, 0.32, [paragraph("Computer Vision Final Project | Hiring-manager presentation", 12, COLORS["muted"])], "Footer")
    )
    slides.append({"xml": slide_xml(shapes), "images": []})

    ids = Ids()
    shapes = (
        title(ids, "Presentation Flow", "Four speakers, one end-to-end computer vision story.")
        + card(ids, 0.75, 1.62, 2.85, 3.15, "Member 1", "Problem and motivation: why exercise counting is useful and difficult.", COLORS["soft_blue"])
        + card(ids, 3.85, 1.62, 2.85, 3.15, "Member 2", "Approach and system design: pose, RGB, TCN, FSM, and routed architecture.", COLORS["soft_green"])
        + card(ids, 6.95, 1.62, 2.85, 3.15, "Member 3", "Demo: squat runtime path and implementation flow.", COLORS["soft_rose"])
        + card(ids, 10.05, 1.62, 2.85, 3.15, "Member 4", "Results, challenges, lessons learned, and closing takeaway.", "FFF7ED")
        + text_box(ids, 1.1, 5.45, 11.2, 0.7, [paragraph("Hiring-manager framing: focus on clear problem definition, evidence-driven architecture decisions, implementation maturity, and honest evaluation.", 17, COLORS["ink"])], "Framing")
        + footer(ids, "Team")
    )
    slides.append({"xml": slide_xml(shapes), "images": []})

    ids = Ids()
    shapes = (
        title(ids, "Problem And Motivation", "Estimate completed exercise repetitions from ordinary workout videos.")
        + text_box(
            ids,
            0.85,
            1.55,
            6.0,
            4.3,
            [
                paragraph("Why it matters", 18, bold=True),
                paragraph("Fitness coaching, physical therapy tracking, remote training, and workout analytics need reliable rep counts.", 15, COLORS["muted"]),
                paragraph("What makes it hard", 18, bold=True),
                paragraph("Viewpoint, scale, lighting, occlusion, body motion, and exercise-specific biomechanics change the visual signal.", 15, COLORS["muted"]),
                paragraph("Engineering goal", 18, bold=True),
                paragraph("Find which representation works best per exercise instead of forcing one universal model.", 15, COLORS["muted"]),
            ],
            "Problem Text",
        )
        + metric_card(ids, 7.5, 1.7, "Input", "Video", "No wearable sensor required", COLORS["blue"])
        + metric_card(ids, 7.5, 3.0, "Output", "Count", "Scalar repetition estimate", COLORS["green"])
        + metric_card(ids, 7.5, 4.3, "Evaluation", "MAE + Within-1", "Error and exact-count reliability", COLORS["rose"])
        + footer(ids, "Member 1")
    )
    slides.append({"xml": slide_xml(shapes), "images": []})

    ids = Ids()
    shapes = (
        title(ids, "Approach And System Design", "A routed architecture selected by validation evidence.")
        + text_box(
            ids,
            0.65,
            1.45,
            3.25,
            4.75,
            [
                paragraph("Design choices", 17, bold=True),
                paragraph("YOLO11n-pose for 17-keypoint pose extraction.", 12, bullet=True),
                paragraph("Normalized pose sequences for shared temporal modeling.", 12, bullet=True),
                paragraph("Squat-specific engineered pose features.", 12, bullet=True),
                paragraph("Frozen ResNet18 RGB features for push-up.", 12, bullet=True),
                paragraph("TCN heads for scalar count regression.", 12, bullet=True),
                paragraph("Exercise label routes to the selected branch.", 12, bullet=True),
            ],
            "Design Bullets",
            fill=COLORS["soft"],
            line=COLORS["line"],
        )
        + picture(ids, 4.15, 1.35, 8.55, 4.85, "rId2", "figure_5_routed_architecture.svg")
        + footer(ids, "Member 2", "figure_5_routed_architecture.svg")
    )
    slides.append({"xml": slide_xml(shapes), "images": ["figure_5_routed_architecture.svg"]})

    ids = Ids()
    shapes = (
        title(ids, "Architecture Exploration", "Experiments show the best representation depends on the exercise.")
        + picture(ids, 0.85, 1.35, 11.65, 5.2, "rId2", "figure_2_architecture_mae_heatmap.svg")
        + footer(ids, "Member 2", "figure_2_architecture_mae_heatmap.svg")
    )
    slides.append({"xml": slide_xml(shapes), "images": ["figure_2_architecture_mae_heatmap.svg"]})

    ids = Ids()
    shapes = (
        title(ids, "Demo Of The Implementation", "The packaged runtime prototype focuses on the strongest branch: squat counting.")
        + text_box(ids, 0.8, 1.55, 2.0, 0.75, [paragraph("Input video", 16, bold=True), paragraph("Workout clip", 10, COLORS["muted"])], "Demo Input", fill=COLORS["soft_blue"], line=COLORS["blue"])
        + text_box(ids, 3.3, 1.55, 2.0, 0.75, [paragraph("YOLO pose", 16, bold=True), paragraph("17 keypoints/frame", 10, COLORS["muted"])], "Demo Pose", fill=COLORS["soft"], line=COLORS["line"])
        + text_box(ids, 5.8, 1.55, 2.0, 0.75, [paragraph("Squat features", 16, bold=True), paragraph("Knee flexion, hip drop", 10, COLORS["muted"])], "Demo Features", fill=COLORS["soft"], line=COLORS["line"])
        + text_box(ids, 8.3, 1.55, 2.0, 0.75, [paragraph("FSM / TCN", 16, bold=True), paragraph("Temporal count model", 10, COLORS["muted"])], "Demo Model", fill=COLORS["soft"], line=COLORS["line"])
        + text_box(ids, 10.8, 1.55, 1.7, 0.75, [paragraph("Count", 16, bold=True), paragraph("Predicted reps", 10, COLORS["muted"])], "Demo Count", fill=COLORS["soft_green"], line=COLORS["green"])
        + line_shape(ids, 2.8, 1.92, 3.3, 1.92, COLORS["muted"])
        + line_shape(ids, 5.3, 1.92, 5.8, 1.92, COLORS["muted"])
        + line_shape(ids, 7.8, 1.92, 8.3, 1.92, COLORS["muted"])
        + line_shape(ids, 10.3, 1.92, 10.8, 1.92, COLORS["muted"])
        + text_box(
            ids,
            0.95,
            3.05,
            5.2,
            2.8,
            [
                paragraph("What the demo shows", 18, bold=True),
                paragraph("Video is converted into pose-derived temporal signals.", 13, bullet=True),
                paragraph("FSM provides an interpretable baseline with movement states.", 13, bullet=True),
                paragraph("TCN learns the mapping from feature sequence to repetition count.", 13, bullet=True),
                paragraph("The live path is squat-only; broader routing is validated offline.", 13, bullet=True),
            ],
            "Demo bullets",
        )
        + text_box(
            ids,
            6.75,
            3.25,
            5.45,
            2.35,
            [
                paragraph("Demo narrative", 18, bold=True),
                paragraph("The model is not judging one frame at a time. It uses motion across the sequence, which is essential for counting repetitions.", 16, COLORS["muted"]),
            ],
            "Demo narrative",
            fill=COLORS["soft"],
            line=COLORS["line"],
        )
        + footer(ids, "Member 3")
    )
    slides.append({"xml": slide_xml(shapes), "images": []})

    ids = Ids()
    shapes = (
        title(ids, "Final Routed Results", "Reportable branches with 95% bootstrap confidence intervals.")
        + picture(ids, 0.8, 1.35, 11.75, 5.15, "rId2", "figure_1_routed_performance_ci.svg")
        + footer(ids, "Member 4", "figure_1_routed_performance_ci.svg")
    )
    slides.append({"xml": slide_xml(shapes), "images": ["figure_1_routed_performance_ci.svg"]})

    ids = Ids()
    shapes = (
        title(ids, "Metric Tradeoff And Model Selection", "The selected model is not always the lowest-MAE model.")
        + picture(ids, 0.75, 1.35, 11.8, 5.15, "rId2", "figure_3_mae_within1_tradeoff.svg")
        + footer(ids, "Member 4", "figure_3_mae_within1_tradeoff.svg")
    )
    slides.append({"xml": slide_xml(shapes), "images": ["figure_3_mae_within1_tradeoff.svg"]})

    ids = Ids()
    shapes = (
        title(ids, "Challenges And Lessons Learned", "Complexity helped only when it matched the visual signal.")
        + text_box(
            ids,
            0.75,
            1.45,
            3.55,
            4.95,
            [
                paragraph("Challenges", 17, bold=True),
                paragraph("Pose noise and occlusion affect keypoint reliability.", 12, bullet=True),
                paragraph("Small validation subsets create wide confidence intervals.", 12, bullet=True),
                paragraph("Exercise motion patterns require different representations.", 12, bullet=True),
                paragraph("MAE alone can hide poor exact-count reliability.", 12, bullet=True),
                paragraph("Lessons", 17, bold=True),
                paragraph("Start from the data signal before adding model complexity.", 12, bullet=True),
                paragraph("Keep negative experiments; they justify architecture decisions.", 12, bullet=True),
            ],
            "Lessons bullets",
            fill=COLORS["soft"],
            line=COLORS["line"],
        )
        + picture(ids, 4.55, 1.35, 7.85, 5.05, "rId2", "figure_4_per_exercise_mae_comparison.svg")
        + footer(ids, "Member 4", "figure_4_per_exercise_mae_comparison.svg")
    )
    slides.append({"xml": slide_xml(shapes), "images": ["figure_4_per_exercise_mae_comparison.svg"]})

    ids = Ids()
    shapes = (
        title(ids, "Closing Takeaway", "An evidence-driven computer vision system, not just a model leaderboard.")
        + text_box(
            ids,
            0.95,
            1.55,
            6.0,
            3.9,
            [
                paragraph("What we built", 19, bold=True),
                paragraph("End-to-end workflow: data preparation, pose extraction, feature construction, model training, evaluation, and demo.", 14, bullet=True),
                paragraph("Architecture evidence: pose, RGB, FSM, TCN, Transformer, and fusion were compared.", 14, bullet=True),
                paragraph("Final result: exercise-dependent routed design plus a working squat runtime prototype.", 14, bullet=True),
            ],
            "Closing bullets",
        )
        + metric_card(ids, 7.55, 1.6, "Squat", "2.14 MAE", "Dedicated pose TCN", COLORS["blue"])
        + metric_card(ids, 7.55, 2.95, "Pull-up", "4.61 MAE", "Shared pose TCN", COLORS["green"])
        + metric_card(ids, 7.55, 4.3, "Push-up", "6.60 MAE", "RGB ResNet18 + TCN", COLORS["rose"])
        + text_box(ids, 1.0, 6.15, 11.2, 0.45, [paragraph("Main message: exercise repetition counting is best handled as exercise-dependent routing, because the reliable visual signal changes by movement type.", 16, COLORS["ink"], bold=True)], "Main message", fill=COLORS["soft_green"], line=COLORS["green"])
        + footer(ids, "Team")
    )
    slides.append({"xml": slide_xml(shapes), "images": []})

    return slides


def content_types(slide_count: int) -> str:
    slide_overrides = "\n".join(
        f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="svg" ContentType="image/svg+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{slide_overrides}
</Types>
"""


def presentation_xml(slide_count: int) -> str:
    slide_ids = "\n".join(
        f'    <p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>
{slide_ids}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_CX}" cy="{SLIDE_CY}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle>
    <a:defPPr><a:defRPr lang="en-US"/></a:defPPr>
  </p:defaultTextStyle>
</p:presentation>
"""


def presentation_rels(slide_count: int) -> str:
    relationships = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml")]
    for i in range(1, slide_count + 1):
        relationships.append((f"rId{i + 1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i}.xml"))
    relationships.append((f"rId{slide_count + 2}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"))
    return rels_xml(relationships)


def root_rels() -> str:
    return rels_xml(
        [
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]
    )


def core_props() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Exercise Repetition Counting From Video</dc:title>
  <dc:subject>Computer vision project presentation</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_props(slide_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Company>Computer Vision Final Project</Company>
</Properties>
"""


def slide_master() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
"""


def slide_layout() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def theme() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Exercise Counting Theme">
  <a:themeElements>
    <a:clrScheme name="Custom">
      <a:dk1><a:srgbClr val="17212B"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="52606D"/></a:dk2>
      <a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="4477AA"/></a:accent1>
      <a:accent2><a:srgbClr val="228833"/></a:accent2>
      <a:accent3><a:srgbClr val="CC6677"/></a:accent3>
      <a:accent4><a:srgbClr val="D97706"/></a:accent4>
      <a:accent5><a:srgbClr val="4F46E5"/></a:accent5>
      <a:accent6><a:srgbClr val="0891B2"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="6D28D9"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos">
      <a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
      <a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Custom">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""


def write_pptx() -> None:
    slides = build_slides()
    image_names = sorted({image for slide in slides for image in slide["images"]})

    with zipfile.ZipFile(PPTX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("docProps/core.xml", core_props())
        zf.writestr("docProps/app.xml", app_props(len(slides)))
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master())
        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            rels_xml(
                [
                    ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
                    ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
                ]
            ),
        )
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout())
        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")]),
        )
        zf.writestr("ppt/theme/theme1.xml", theme())

        for idx, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", str(slide["xml"]))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels(list(slide["images"])))

        for image in image_names:
            image_path = OUT_DIR / image
            zf.write(image_path, f"ppt/media/{image}")


def main() -> None:
    write_pptx()
    print(PPTX_PATH)


if __name__ == "__main__":
    main()
