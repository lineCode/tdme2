TDME2
=====

    - What is it?
        - ThreeDeeMiniEngine2 is a lightweight 3D engine including tools suited for 3D games development using C++11

    - What is already working
        - 3d engine
            - model parsing
                - WaveFront OBJ
                - DAE parsing with skinned meshes and animations
                    - group names/ids must not have whitespace characters
                    - requires baked matrices
                - FBX via FBX SDK
                - TDME Model file format
                    - this is a much more efficient model file format for TDME
                    - can be read and written
                - DAE and WaveFront OBJ files require triangulated meshes for now
            - object transformations
                - scaling
                - rotations
                - translation
            - animations
                - supports base animation and animation overlays
                - supports attaching objects to bones of another objects
            - color effects on objects
                - via shader
                - color addition
                - color multiplication
            - lighting via shaders
                - supports phong lighting
                - supports phong shading on GL3, GL2
                - supports gouraud shading on GLES2
                - supports diffuse mapping on GL3, GL2, GLES2
                - supports specular shininess mapping on GL3
                - supports normal mapping on GL3
            - dynamic shadows via shaders
            - particle system which
              - is object based
              - or point based
              - and supports
                - basic/point emitter
                - sphere emitter
                - bounding box emitter
                - circle on plane emitter
                - ...
            - camera control
              - set up look from, look at, up vector can be computed
              - frustum culling
                - oct tree like partitioning from 64mx64mx64m up to 16mx16mx16m
            - object picking
            - supports offscreen instances
                - rendering can be captured as screenshot
                - rendering can be used (in other engine instances) as diffuse texture
            - screenshot ability
            - multiple renderer
              - GL2, GL3(core) and GLES2
        - physics
            - discrete collision detection
                - sphere
                - capsule
                - axis aligned bounding boxes
                - oriented bounding boxes
                - triangle
                - convex mesh
            - rigid body simulator
              - broadphase collision detection
                  - uses oct tree like partitioning from 64mx64mx64m up to 16mx16mx16m
                  - additionally sphere <> sphere test
              - narrowphase collision detection
              - collision filtering by type
              - sleeping technology
        - path finding
            - uses A*
            - is paired with physics world to determine if a "cell" is walkable
            - optional custom walkable test
        - 3d audio
            - decoder
              - ogg vorbis decoder
            - audio entities
              - streams
              - sounds
        - GUI system
            - borrows some ideas from Nifty-GUI regarding XML and layouting
            - borrows some ideas from AngularJS like
                - all nodes are in the GUI node tree and can be made visible or unvisible depending on conditions
            - adds some improvements like
                - support auto keyword with nodes width and height attributes
            - supported primitive nodes from which compounds are built of
                - element
                - image
                - input
                - layout
                - panel
                - scrollbars
                - space
                - text
            - supported compound elements
                - button
                - checkbox
                - dropdown
                - input
                - radio button
                - scrollarea both
                - scrollarea horizontal
                - scrollarea vertical
                - selectbox
                - selectbox multiple
                - tabs
            - supports position and color based effects
        - Networking module, which consists of
            - UDP server
                - n:m threading model with non blocked IO via kernel event mechanismns(epoll, kqueue or select)
                - supports safe messages with acknowledgment and automatic resending
                - support fast messages
                - can be used in a heavy multithreaded environment (the networking module is thread safe)
            - UDP client
                - has single thread with a simple threadsafe API
                - supports all features required by UDP server

    - What does it (maybe still) lack
        - animation blending
        - physics
          - bounding volume hierarchies
          - multiple bounding volumes for a rigid body
          - rag doll / joints / springs
        - example games
        - documentation

    - What is WIP
        - rigid body simulator(needs to be updated to newer "ReactPhysics3D 0.5")
        - GUI system port needs to be finished(Memory Management and other minor things)
        - Logic documentation/comments need to be imported from TDME(-JAVA)

    - Technology
        - designed for simple multi threading
            - 3d engine uses one thread for now
            - physics or game mechanics can run in a separate thread(s)
        - uses 
            - GLUT
            - OpenGL
            - OpenAL
            - glew
            - Vorbis/OGG,
            - JsonBox
            - zlib
            - libpng
            - tinyxml
            - pthreads
            - (collision resolving of) ReactPhysics3D
            - FBXSDK
            - V-HACD
        - targeted platforms and its current state
            - Windows/MINGW(port completed)
            - Linux(port completed)
            - Mac Os X(port completed)
            - Android(port pending)
            - iOS(port pending)

    - Tools
        - TDME Model Editor, see README-ModelEditor.md
        - TDME Particle System Editor, see README-ParticleSystemEditor.md
        - TDME Level Editor, see README-LevelEditor.md

    - Links
        - <youtube link here>

    - References
        - "game physics - a practical introduction" / Kenwright
        - "real-time collision detection" / Ericson
        - "ReactPhysics3D" physics library, http://www.reactphysics3d.com 
        - the world wide web! thank you for sharing

    - Other credits
        - Dominik Hepp
        - Mathias Wenzel
        - Sergiu Crăiţoiu
        - Kolja Gumpert
        - others
       
