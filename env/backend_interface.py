'''
This file is used to import the correct environment backend based on the build type.
'''

if __debug__:
    from build.Debug.envbackend import envbackend as env
else:
    from build.Release.envbackend import envbackend as env