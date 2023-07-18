import time
import edgeiq
import cv2
from datetime import datetime



def main():
    # obj_detect = edgeiq.ObjectDetection("dougrien/dc-person-detection-all-20220510-300x300-aug")
    obj_detect = edgeiq.ObjectDetection("dougrien/jamba-01-person-detect-mob-e100-b8-x300-20230129")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    kalman_tracker = edgeiq.KalmanTracker(deregister_frames=20, max_distance=100)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    zones = edgeiq.ZoneList("zone_config.json")
    door = zones.get_zone('Zone A')
    door_box = edgeiq.create_bounding_box_from_zone(door)
    record_time = {}
    current_time_dict = {}
    first_time_dict = {}

    try:
        # with edgeiq.FileVideoStream("jamba-kaiser-20230121-2.mp4") as video_stream, \
        with edgeiq.IPVideoStream('http://root:pass@192.168.0.91/axis-cgi/mjpg/video.cgi') as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            while True:
                text = ' '

                image = video_stream.read()

                # my_zones = edgeiq.load_zones_from_config('zone_config.json')

                results = obj_detect.detect_objects(image, confidence_level=.4)
                objects = kalman_tracker.update(results.predictions)

                image = zones.markup_image_with_zones(image, fill_zones=True, color=(255, 0, 0))
                # image = edgeiq.markup_image(image, results.predictions, colors=((0, 0, 0), (0, 255, 0), (0, 0, 0)))
                # check1 = zones.get_zones_for_prediction(results.predictions)
                # test = zones.get_results_for_zone(results, 'Zone A')
                # check = zones.filter_results_by_zone(objects)
                # name = zones.get_zone_names()
                #
                # print(check, 'CHECK')
                # print(name, 'name')
                # print(type(check))
                # print(results.predictions)
                # print(len(results.predictions), "length")

                # print(objects.items())
                # print(len(objects.items()), "after clear length")
                try:
                    if len(results.predictions) > 0:
                        results.predictions.clear()

                        for (object_id, trackable_prediction) in objects.items():

                            # if len(trackable_prediction) > 0:
                            # if edgeiq.Zone.check_object_detection_prediction_within_zone(trackable_prediction):
                                if door_box.compute_overlap(trackable_prediction.box):
                                # if zones.get_zones_for_prediction(results.predictions):
                                    # print(trackable_prediction.box.compute_overlap(door_box),' GOT IT')
                                    # print(len(results.predictions))
                                    if object_id not in current_time_dict.keys():
                                        first_time_dict[object_id] = time.strftime('%H.%M.%S')
                                    # print('running')

                                    # image = edgeiq.markup_image(image, results.predictions, colors = ((0, 0, 0), (0,255,0), (0,0,0)))
                                    # image = edgeiq.markup_image(image, results.predictions, colors=obj_detect.colors)
                                    # Generate text to display on streamer
                                    text = ["Model: {}".format(obj_detect.model_id)]
                                    text.append(
                                            "Inference time: {:1.3f} s".format(results.duration))
                                    text.append("Objects:")

                                    current_time_dict[object_id] = time.strftime('%H.%M.%S')

                                    new_label = 'Person {} '.format(object_id)
                                    trackable_prediction.label = new_label
                                    # results.predictions.append("{}:".format(trackable_prediction))
                                    results.predictions.append(trackable_prediction)

                                    text.append("{}:".format(object_id))

                    else:
                        for object_id in first_time_dict.keys():
                            if object_id in current_time_dict.keys():
                                last_time = current_time_dict.get(object_id)
                                last_time = datetime.strptime(last_time, '%H.%M.%S')
                                first_time = first_time_dict.get(object_id)
                                first_time = datetime.strptime(first_time, '%H.%M.%S')
                                print('waiting time for', object_id, 'is', last_time - first_time)
                        # print('clear frame')
                        first_time_dict.clear()
                        current_time_dict.clear()

                except Exception as ex:
                    print('ERROR', ex)

                image = edgeiq.markup_image(image, results.predictions, colors=((0, 0, 0), (0, 255, 0), (0, 0, 0)))

                streamer.send_data(image, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
