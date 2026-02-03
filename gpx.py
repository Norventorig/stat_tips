# import gpxpy
#
# with open("track.gpx", "r", encoding="utf-8") as f:
#     gpx = gpxpy.parse(f)

# points = []
#
# for track in gpx.tracks:
#     for segment in track.segments:
#         for point in segment.points:
#             points.append(
#                 (
#                     point.latitude,
#                     point.longitude,
#                     point.elevation,
#                     point.time
#                 )
#             )
