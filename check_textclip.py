import inspect
from moviepy.video.VideoClip import TextClip
print('Signature:', inspect.signature(TextClip))
print('\nParameters:')
for name,param in inspect.signature(TextClip).parameters.items():
    print(name, param)
print('\nSource snippet:\n')
src = inspect.getsource(TextClip)
print('\n'.join(src.splitlines()[:120]))
