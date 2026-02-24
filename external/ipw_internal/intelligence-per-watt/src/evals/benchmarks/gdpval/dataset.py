"""
GDPval dataset helpers.
- Downloads tasks from HuggingFace
- Applies the official prompt suffix
- Normalises reference file paths
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from datasets import load_dataset

HF_DATASET_PATH = "openai/gdpval"


DEFAULT_PROMPT_SUFFIX = """\nFiles can be found using the tools provided. Packages are installed in the environment, including:
libreoffice
aspose-words==25.8.0
av==11.0.0
cadquery==2.4.0
cadquery-ocp==7.7.0
pedalboard==0.9.9
pyloudnorm==0.1.1
srt==3.5.3
xlrd==2.0.1

Write your files in a new folder named `deliverable_files`. We will also grade your final message as part of the deliverable."""
# Optional: expose the longer capabilities suffix if you ever need it
PROMPT_ELICIT_CAPABILITIES_SUFFIX = """Special characters
- Never use the character - (U+2011), since it will render poorly on some people’s computers. Instead, always use - (U+002D) instead.
- Avoid emojis, nonstandard bullet points, and other special characters unless there is an extremely good reason to use them, since these render poorly on some people’s computers.

Graphics embedded within PDFs/slides
- Make sure that any diagrams or plots are large enough to be legible (though not so large that they are ugly or cut off). In most cases they should be at least half the page width.
- Plots and charts to visualize data are good. Simple graphics (like a flowchart with arrows) are good. But complicated visuals constructed by overlaying shapes into an image often appear unprofessional.

PDFs
- Always use LibreOffice to create the PDF (it must be LibreOffice! If LibreOffice is not installed, you can install it yourself). Other libraries sometimes show weird artifacts on some computers.

Fonts
- Always use fonts which are available across all platforms. We recommend Noto Sans / Noto Serif unless there is an extremely good reason to use something else.
- If you must use another font, embed the font in the pptx/word/etc doc.

Deliverable text
- Do not link to submitted files in the deliverable text (links are not supported on the interface where these will be viewed).
- Ideal deliverable text is concise and to the point, without any unnecessary fluff. 4 sentences max.
- Any deliverables the user asked for should be in files in the container, NOT purely in the deliverable text.
- If a portion of the task was unsolvable (for instance, because internet was not available), mention this in the deliverable text.
- Your submission should be complete and self-contained. Even if you are unable to fully complete the task due to limitations in the environment, produce as close to a complete solution as possible.

Verbosity
- Always be clear and comprehensive, but avoid extra verbosity when possible.

Filetypes
- If the prompt does not request a specific filetype, use standard filetypes like PDF, PPTX, DOCX, XLSX, MP4, ZIP, etc.

Video files (mp4, mov)
- Extract a string of images from the video files and check the images to see whether the visual elements are corrupted.

Mandatory formatting checks
- Before you submit your deliverable, you MUST perform the following mandatory formatting checks. Take your time, do these thoroughly, they are extremely important!

STEP 1: Convert all visual deliverables to PNGs using LibreOffice. This includes pptx, docx, pdf, xlsx, etc. Convert it so that each page or slide is a separate PNG. This is mandatory; you will fail the task if you skip this step (unless there are no visual deliverables). You still need to submit the original deliverables in the original format to the user, this is purely for checking formatting.
STEP 2: Display the PNGs. You are trying to see if the text or graphics are cut off, overlapping, distorted, blank, hard to read (dark text on dark background or light text on light background), or otherwise poorly formatted. Look at each image thoroughly, zoom in if you need to see more closely. Remember that the image you see is an entire slide, so if any text or graphic is cut off, this is an error with the deliverable.
STEP 3: Programmatic formatting checks. For highly visual submissions (e.g. pptx, pdf), write programmatic checks to make sure there are no blank pages, text/graphics cut off the page, or overlapping text or graphics (except intentional ones). Also check that if there is a page or slide limit, it is respected.
STEP 4: Summarize the prompt’s deliverable instructions, and match that to the portion of the deliverable that addresses it.
STEP 5: Right before submitting, check that the deliverables you have produced are exactly what you want to submit: deliverables should contain exactly the files you want to submit, with no extra files. Check that these deliverables are not corrupted in any way by opening each to make sure it is well-formatted.

If any of these checks reveal a formatting issue, fix them and go through steps 1-5 again. Take your time, be thorough, remember you can zoom in on details.

Finally - on the last line of your output text, add CONFIDENCE[XX], where XX is an integer between 0 and 100, inclusive, indicating your confidence that the submission is correct, follows instructions, and is well-formatted.

CONFIDENCE[93]"""


@dataclass
class GDPvalSample:
    task_id: str
    prompt: str
    files: Dict[str, str]  # {container_relative_path: download_url}
    metadata: Dict[str, object]


def remove_middle_stem(file_path: str) -> str:
    """
    Reference files are published as reference_files/<hash>/<filename>.
    GDPval instructions refer to them without the hash, so we normalise.
    """
    parts = file_path.split("/")
    if len(parts) == 3 and parts[0] == "reference_files":
        return f"{parts[0]}/{parts[2]}"
    return file_path


def load_gdpval_samples(
    *,
    shuffle: bool = False,
    limit: Optional[int] = None,
) -> Iterator[GDPvalSample]:
    """
    Stream GDPval samples from HuggingFace, adding the prompt suffix and
    normalising file paths.
    """
    ds = load_dataset(
        HF_DATASET_PATH,
        split="train",
        streaming=False,
    )

    rows = ds.shuffle(seed=0) if shuffle else ds
    if limit is not None:
        rows = rows.select(range(min(limit, len(ds))))  # type: ignore[arg-type]

    for row in rows:
        files = {
            remove_middle_stem(path): url
            for path, url in zip(row["reference_files"], row["reference_file_urls"])
        }
        prompt = f"{row['prompt']}{DEFAULT_PROMPT_SUFFIX}"
        yield GDPvalSample(
            task_id=row["task_id"],
            prompt=prompt,
            files=files,
            metadata={
                "sector": row["sector"],
                "occupation": row["occupation"],
                "reference_file_hf_uris": row["reference_file_hf_uris"],
            },
        )