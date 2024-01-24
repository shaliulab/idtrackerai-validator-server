import os.path
import cv2

chunksize=45000

class GroupBehaviorReader:

    @classmethod
    def from_basedir(cls, basedir):
        readers={}        
        
        flyhostel_single_animal_root=os.path.join(basedir, "flyhostel", "single_animal")
        if not os.path.exists(flyhostel_single_animal_root):
            return readers
        

        identities=os.listdir(flyhostel_single_animal_root)
        if len(identities) == 0:
            return readers

        for identity in identities:
            readers[int(identity)]=BehaviorReader(os.path.join(flyhostel_single_animal_root, identity))
        return readers
    


class BehaviorReader:

    def __init__(self, basedir, with_behavior=False):
        self._basedir=basedir
        self._last_frame_idx=None
        self._last_chunk=None
        self._cap=None
        self._with_behavior=with_behavior
        

    def _init_video_capture(self, video_path, chunk, frame_idx):
        self._cap = cv2.VideoCapture(video_path)
        self._cap.set(7, frame_idx)
        self._last_frame_idx=frame_idx
        self._last_chunk=chunk

    def get_image(self, frame_number):

        chunk = frame_number // chunksize
        frame_idx = frame_number % chunksize

        video_path = os.path.join(self._basedir, str(chunk).zfill(6) + ".mp4")

        if self._cap is None:
            self._init_video_capture(video_path, chunk, frame_idx)


        elif chunk == self._last_chunk:
            if (self._last_frame_idx + 1)!=frame_idx:
                self._cap.set(7, frame_idx)
   
            ret, frame = self._cap.read()
            self._last_frame_idx+=1
        
        
        else:
            self._cap.release()

            self._init_video_capture(video_path, chunk, frame_idx)
            ret, frame = self._cap.read()
            self._last_frame_idx+=1

        return frame

