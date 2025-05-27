# from odf.opendocument import load, OpenDocumentText
# from odf.text import P
#
# def merge_odf_files(input_files, output_file):
#     # Create a new ODF document for output
#     output_doc = OpenDocumentText()
#
#     for file in input_files:
#         # Load each ODF file
#         doc = load(file)
#
#         # Get content from the loaded document and add to output document
#         for elem in doc.getElementsByType(P):  # Only paragraphs, for example
#             output_doc.text.addElement(elem)
#
#     # Save the merged content to the new output file
#     output_doc.save(output_file)
#
# # Specify the files to merge and the output file
# # input_files = ["file1.odt", "file2.odt", "file3.odt"]
# output_file = "merged_document.odt"
# merge_odf_files(input_files, output_file)
