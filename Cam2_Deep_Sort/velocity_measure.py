import numpy as np
import scipy.io
import os
import pickle
import pdb
import codecs

def compute_vel(box_vel, det, frame, s_x, s_y, y_min, y_max, x_limit, H, location, V0, video_id):
    vel = V0

    for i, w in enumerate(box_vel):
        c = np.array([(det[0] + det[2])/2.0 , det[3] - y_min])
        v = box_vel

        if i == 0:
            f = frame
        else:
            k = 1.5
		#implementation of Homography matrix
            v_x_trans = (((H[0][0]*v[0] + H[0][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                         (H[0][0] * c[0] + H[0][1] * c[1] + H[0][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                        ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2])**2))
            v_y_trans = (((H[1][0]*v[0] + H[1][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                         (H[1][0] * c[0] + H[1][1] * c[1] + H[1][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                        ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2])**2))

            instant_vel = np.sqrt(sum([(v_x_trans * s_x)**2, (v_y_trans * s_y )**2]))  #displacement
            t_delta = 1  #frame rate
            vi = instant_vel / t_delta * 30 * 9 / 4

            if location in ['Loc']:    #may need modification
            	if det[2] > x_limit :
            		if location=='Loc':
            			s1 = 1.0
            			s2 = 1.0 * 1.1
            		else:
                		s1 = 0.7 * 1.1
                		s2 = 1.0 * 1.1

            	elif location=='Loc':
            			s1 = 1.0 * 1.1
            			s2 = 1.4 * 1.1
            	else:
            		s1 = 1.0 * 1.1
            		s2 = 1.8 * 1.1

            	a = (s2-s1) / (y_max - y_min)
            	b = s1
            	s = a * c[1] + b
            	v_estimate = (vel + vi * s ) / 2.0
            else:
            	v_estimate = (vel + vi) / 2.0
            f = frame
            vel = v_estimate

    return vel


def compute_vel_back_up(box_vel, det, frame, s_x, s_y, y_min, y_max, x_limit, H, location, V0, video_id):
    ##det: (x_1, y_1, x_2, y_2)

    vel = [V0]


    for i, v in enumerate(box_vel):

        ##print(i)
        ##print(v)

        c = np.array([(det[i][0] + det[i][2])/2.0 , det[i][3] - y_min])

        if i == 0:
            f = frame[i]
        else:
            k = 1.5
		#implementation of Homography matrix
            v_x_trans = (((H[0][0]*v[0] + H[0][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                         (H[0][0] * c[0] + H[0][1] * c[1] + H[0][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                        ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2])**2))
            v_y_trans = (((H[1][0]*v[0] + H[1][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                         (H[1][0] * c[0] + H[1][1] * c[1] + H[1][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                        ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2])**2))

            instant_vel = np.sqrt(sum([(v_x_trans * s_x)**2, (v_y_trans * s_y )**2]))  #displacement
            t_delta = frame[i] - f  #frame rate
            vi = instant_vel / t_delta * 30 * 9 / 4

            if vi<=3.0:  #outliers?
            	vi = 0.0

            if location in ['Loc1','Loc2']:    #may need modification
            	if det[i][2] > x_limit :
            		if location=='Loc1':
            			s1 = 1.0
            			s2 = 1.0 * 1.1
            		else:
                		s1 = 0.7 * 1.1
                		s2 = 1.0 * 1.1

            	elif location=='Loc1':
            			s1 = 1.0 * 1.1
            			s2 = 1.4 * 1.1
            	else:
            		s1 = 1.0 * 1.1
            		s2 = 1.8 * 1.1

            	a = (s2-s1) / (y_max - y_min)
            	b = s1
            	s = a * c[1] + b


            	if location == 'Loc1':  #may need modification
            		if video_id == '1':
            			if det[i][1] < 800 and det [i][1] > 550 :
            				if det[i][2] > x_limit:
            					vi = vi * 0.9
            	v_estimate = (vel[i - 1] * i + vi * s ) / (i + 1)
            elif location == 'Loc3':
            	if det[i][2] > x_limit :

            		s1 = 0.8
            		s2 = 0.95
            		a = (s2-s1) / (y_max - y_min)
            		b = s1

            	else:
            		s1 = 1.3
            		s2 = 1.6
            		a = (s2-s1) / (y_max - y_min)
            		b = s1

            	s = a * c[1] + b
            	v_estimate = (vel[i-1] + vi * s) / 2.0
            else:
            	v_estimate = (vel[i-1] + vi) / 2.0
            f = frame[i]
            vel.append(v_estimate)

            #pdb.set_trace()
    return vel



# (s_x, s_y, limit)
info = {'Loc': {'V0': 1, 's_x': 3.2/5, 's_y': 4.0, 'y_min': 300.0, 'y_max':1000.0 , 'L0': 0.2, 'f1': 905.53, 'f2': -71.2 } }

if __name__ == '__main__':
    f = open('./velocity_output/track.txt','w')
    video_index = 1
    item = "Loc"
    s_x = info[item]['s_x']
    s_y = info[item]['s_y']
    f1 = info[item]['f1']
    f2 = info[item]['f2']
    l0 = info[item]['L0']
    H = [[l0,-l0*(f1/f2),0.0],[0.0,1.0,0.0],[0.0,-(1/f2),1.0]]
    x_limit = 1920/2.0
    y_min = info[item]['y_min']
    y_max = info[item]['y_max']
    V0 = info[item]['V0']

    frame = []
    track_ids = []
    detections = []
    velocities = []
    score = []
    data = []

    #f2 = open('./velocity_input/data.txt','r')
    f2 = open('./deep_sort_output.txt','r')

    ##input format

    ## data[0]  data[1]    data[2-5]    data[6-7]    data[8]
    ##  frame   track_id   detections   velocities   score

    for line in f2.readlines():
        data = line.rstrip("\n").split(' ')
        frame.append(int(data[0]))
        track_ids.append(float(data[1]))
        detections.append([float(data[2]),float(data[3]),float(data[4]),float(data[5])])
        velocities.append([float(data[6]),float(data[7])])
        score.append(float(data[8]))
    f2.close()

    # filter out bounding boxes that is not in the cropped image
    detections_filtered = []
    track_ids_filtered = []
    frame_filtered = []
    velocities_filtered = []
    score_filtered = []

    for i in range(len(detections)):  #use different video location data to input relevant data into matrices
        det = detections[i]
        if det[1] > y_min :
            detections_filtered.append(det)
            velocities_filtered.append(velocities[i])
            track_ids_filtered.append(track_ids[i])
            frame_filtered.append(frame[i])
            score_filtered.append(score[i])

    detections_filtered = np.matrix(detections_filtered)
    velocities_filtered = np.matrix(velocities_filtered)
    track_ids_filtered = np.array(track_ids_filtered)
    frame_filtered = np.array(frame_filtered)
    score_filtered = np.array(score_filtered)

    frame_final = []
    track_id_final = []
    score_final = []
    det_final = []
    vel_final = []
    vel_boxes = []

    location = 'Loc'
    video_id = 1

    for tr in np.unique(track_ids_filtered):
        det = detections_filtered[track_ids_filtered == tr]
        det = det.tolist()
        box_vel = velocities_filtered[track_ids_filtered == tr].tolist()
        tr_frame = frame_filtered[track_ids_filtered == tr].tolist()
        box_score = score_filtered[track_ids_filtered == tr].tolist()

        vel = compute_vel_back_up(box_vel, det, tr_frame, s_x, s_y, y_min, y_max, x_limit, H, location, V0, video_id)
        frame_final.extend(tr_frame)
        track_id_final.extend([tr] * len(det))
        vel_final.extend(vel)
        det_final.extend(det)
        score_final.extend(box_score)

    data_final = {}
    data_final['track_ids'] = track_id_final
    data_final['frame_num'] = frame_final
    data_final['detections'] = det_final
    data_final['velocity'] = vel_final
    data_final['score'] = score_final

    print ('Done ' )

    ## output format

    ## frame, track_id, bbox, velocity

    for i in range(0, len(track_id_final)):
    	f.write('{} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n'.format(frame_final[i] + 1, int(track_id_final[i]), det_final[i][0], det_final[i][1], det_final[i][2], det_final[i][3], vel_final[i]))

    print ('Saved ')
    f.close()
