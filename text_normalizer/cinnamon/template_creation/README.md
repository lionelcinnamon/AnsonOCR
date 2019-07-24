# THE PIPELINE

1. Generation
1a. Generate template for users to write in: a PDF with specific filename
1b. Generate configuration files (json files) that help with extraction later

2. Extraction
- Input: PDF with the same page order and page number as the template PDF
- Output: folder of images and json label (advise against having japanese characters in file name, as based on the system, it can cause problem)
- Use the configuration and the PDF to automatically extract labels
- A fast way to mark column key points

# TODO
- Some fixed templates

# CHALLENGES
- The layout analysis is not exactly precise (it might over-segments when the characters in the box have long vertical stroke)