import os
import re
import cv2



######################################################################
# cambiarlo en un futuro y llamar a utils !!!
def sort_list_of_fs_by_ascending_number(list_of_fs, r_pattern = ''):
    '''
    This function sorts a list of files/words by ascending order 
    E.g.(1): sort_list_of_fs_by_ascending_number(['Model_10.inp','Model_1.inp'])
    E.g.(2): sort_list_of_fs_by_ascending_number(['t_1 Model_10.inp','t_2 Model_1.inp'], 'Model_')
    ----------
    list_of_fs: [list of files/words]
    r_pattern: [str/regex] | regex pattern | Def. arg.: ''
    ----------
    the function modifies the list
    '''
    list_of_fs.sort(key=lambda el:int(re.search(f'{r_pattern}(\d+)',el).group(1)))
######################################################################



try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

folder_name = 'pop'
# tr_005 tr_01 tr_02 tr_05 tr_1 tr_2
prefix_name = None # None | prefix1, prefix2, ...
images_path = os.path.join(script_dir, folder_name)
image_names = [f for f in os.listdir(images_path) if f.endswith('jpg') and (f.startswith(prefix_name) if prefix_name else f.startswith('image_'))]

if image_names:
    ######################################################################
    # cambiarlo en un futuro y llamar a utils !!!
    ######################################################################
    sort_list_of_fs_by_ascending_number(image_names)
    im_nun = int(image_names[-1].split("_")[-1].split(".")[0])
    i = im_nun+1
else:
    i = 1

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame.")
        break
    
    cv2.imshow('Presiona espacio para capturar la foto', frame)

    key = cv2.waitKey(2)
    if key == 32:  # Tecla espacio
        if prefix_name is not None:
            image_n = f'{prefix_name}_image_{i}'
        else:
            image_n = f'image_{i}'
        frame = cv2.resize(frame, None, fx= .5, fy= .5)
        cv2.imwrite(f'{images_path}/{image_n}.jpg', frame)
        print(f"Foto guardada como '{image_n}.jpg'")
        i += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

