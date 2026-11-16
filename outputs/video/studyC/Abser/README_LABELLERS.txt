STUDY C LABELLING PACK (Phase 6). Read the protocol first: Phase6_studyC_audit_protocol.md (in this zip).

What this is: 350 pictures of detections the face detector made that the dataset annotation does not cover. You label each one by eye.
These images show real people (ChokePoint dataset, non-commercial research use). Use the pack only for this audit, do not copy or share it, do not upload it anywhere,
and DELETE it after you have sent your label files back.

You need: Python 3 with   pip install opencv-python numpy pandas   and a screen at least 1200 pixels wide.

Unzip the pack into a folder, open a terminal in it and run:

  Stage 1 (the 50 crops all three of us label; work alone, do not discuss crops with anyone yet):
      python video_studyC_audit.py label --who YOURNAME --pack . --stage shared
  Send the file  labels_YOURNAME.csv  to Claude's lab-PC session (via Abser or Umar) when done. Then WAIT for the go-ahead.

  Stage 2 (your own 100 crops, only after the go-ahead):
      python video_studyC_audit.py label --who YOURNAME --pack . --stage own
  Send labels_YOURNAME.csv again (it now holds both stages). Then delete the pack.

YOURNAME is one of: abser, raghib, umar (lower case).

Keys in the window:
  a  live face, annotated person (clearly the same person as one of the reference crops on the right)
  o  live face, other person
  r  reflection or picture of a face (glass door, mirror, screen, poster)
  n  not a face (back of head, body, object)
  u  unclear
  b  back one crop (undo the last label)      q  save and quit

The script saves after every key and resumes where you stopped; run the same command again to continue.
One look, one key. Do not zoom or look for more context than the window shows.
