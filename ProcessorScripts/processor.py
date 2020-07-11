import sys, face_alignment
from skimage import io


def main(argv):
    input_img = argv[0]

    # Generate the text file for face alignment
    face_alignment_data = get_alignment(input_img)
    # Remove the file extension and replace it with .txt
    alignment_data_file_path = ".".join(input_img.split(".")[:-1]) + ".txt"
    with open(alignment_data_file_path, 'w') as file:
        file.write(face_alignment_data)

    # Now send that to Deep


def get_alignment(input_img):
    # We don't have a server with a GPU so this has to run on CPU
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")
    input = io.imread(input_img)
    output = fa.get_landmarks(input)[-1]

    # Convert that into a TSV
    output = ""
    for data_point in output:
        output += data_point[0]
        output += "\t"
        output += data_point[1]
        output += "\n"

    return output.rstrip()


if __name__ == "__main__":
    main(sys.argv[1:])
