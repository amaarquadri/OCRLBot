# Setup Instructions for Windows
1. Setup RLBot according to [this tutorial](https://www.youtube.com/playlist?list=PL2MGDOTjPtl8fuoXmqxTmASW1ZtrPEXQ2), but use this fork of GoslingUtils instead.
2. Go to Steam -> Library -> Rocket League -> Workshop, search for "Lethamyr's Neon Heights Rings" and install it
3. Go to `C:\Program Files (x86)\Steam\steamapps\common\rocketleague\TAGame\CookedPCConsole`
4. Rename `Labs_CirclePillars_P.upk` to `Labs_CirclePillars_P.upkbackup`
6. Copy `C:\Program Files (x86)\Steam\steamapps\workshop\content\252950\2468375059\LethNeonHeightsRings.udk` to `C:\Program Files (x86)\Steam\steamapps\common\rocketleague\TAGame\CookedPCConsole\Labs_CirclePillars_P.upk` (note the change in the file extension)
7. Launch RLBot, Click Add -> Load Cfg File, and select `src/ConvexMPCBot.cfg`
8. Set the map to Pillars, and add Unlimited Match Length and Unlimited Boost Amount to the Mutators.
9. Launch

For Epic Games the relevant folder should be `C:\Epic Games\games\rocketleague\TAGame\CookedPCConsole`. Instead of step 2, you can use the `LethNeonHeightsRings.udk` file in this repo.

# Collecting a Reference Trajectory
1. Add a Human alongside OCRLBot in the match on opposite teams when setting up the RLBot GUI.
2. Modify the OCRLBot code to log the foes' state at every timestep.
3. Unsure how to log the foes' control inputs.
