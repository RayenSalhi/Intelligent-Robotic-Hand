import mediapipe as mp #libraries hedhom lkol lezmk tinstallehom aal visual studio 9ball matekhdem bch yaarefhom wekhdem bel visual studio 2022 mech bel vs code 3ossou rzin chwy  
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import serial


# lport elli mahtouta fih lcart na 3andi com 4
port = 'COM4'

# tasna3 serial object bech tab3th aalih e donnée elli hou l carta arduino
arduino = serial.Serial(port=port, baudrate=9600, timeout=0.01)


#  lfnction hedhi ta3taha lista taa les angles taa kol sbo3 wtab3thha lel arduino aal serial point adheka aalh estaamlna spi protocol fel carta SPI +"serial perephiral interface"

def set_angles(angles):
   
    
    msg = ''
    
    #tappendi 0 aal isar lel les angles elli fehom ken zoz ar9am bch fel code taa larduino yabdou les angles lkoll kif kif
    for angle in angles:
        a = str(angle)
        while len(a) < 3:
            a = '0' + a
        msg += a
    
    # hedhom des symboles fi awel w ekhr l msg bech taaref l carta wa9teh tabda trecievi tnajm thot ay symbole 
    msg = '<' + msg + '>'

    # hna tab3th fel msg aala serial  
    print("Sending: ", msg)    
    for c in msg:
        arduino.write(bytes(c, 'utf-8'))
    
    # ta9ra e data elli teekteb fehom larduino aala serial 
    data = arduino.readline()
    print("Receiving: ", data)


#
#   tmappi valeur men range l range 
#
def translate(value, leftMin, leftMax, rightMin, rightMax):

    # Figure out how the size of each range
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Normalize the value 
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Scale the value 
    return rightMin + (valueScaled * rightSpan)


#  lfonction hedhi tehseb les angles mtaa kol sbo3  

def compute_finger_angles(image, results, joint_list):

    angles = []

    for hand in results.multi_hand_landmarks:
        for i, joint in enumerate(joint_list):
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

            rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(rad*180.0/np.pi)

            if angle > 180:
                angle = 360 - angle
            
            if i == 0:
                angle = np.interp(angle,[90,180],[0, 200])
                angle = min(180, angle)
            else:
                angle = np.interp(angle,[30,180],[0, 180])
                angle = min(180, angle)

            angles.append(int(angle))
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)
    return image, angles




# Setup mediapipe bch edetecty l key points w torsmhom aal camera
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Stinitializi l camera 
cap = cv2.VideoCapture(0)


# takhtar l poinet elli bch tekhdem behom bch tehseb langle taa lfingers elli houma 3 point elli fel mfasel taa swaba3
joint_list = [ [4, 3, 2], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]

#hedha algorithm taa filtrage d'image mtaa lmediapipe 3amlettou lgoogle matbadel fih chy si lcamera maadch tnajm tcaptilk idk
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():

        # capture webcam image and convert it from BGR to RGB
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip the image horizontally
        image = cv2.flip(image, 1)
        
        # detect the hand with MediaPipe
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # render detections
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                
                # render the detected landmarks 
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(0, 0, 155), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # lhna 3ayetna lele fonction elli tehsb les angles w aainaha limage baad ma aamalna lfiltrage mte3ha w kharejna menha ell keypoints elli hachtna behom
            image, angles = compute_finger_angles(image, results, joint_list)
            set_angles(angles) # aayetna lel fonction elli tab3th les angles lel carta 
        
        # convert the image back to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Tracking', image)

       

        # end the loop if <q> is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break




# close webcam feed
cap.release()
#out.release()
cv2.destroyAllWindows()
