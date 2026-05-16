# How to Load Data from Projects in DreamEdit3D Tab

## ✅ New Feature: Load from Current Project

The DreamEdit3D Training tab now has **easy project integration**!

## 🎯 Two Ways to Load Data

### Method 1: Load from Current Project (Recommended) ⭐

**This is the easiest way!**

1. **Select/Create a project** in the Project Management section at the top
2. **Run SAM segmentation** in the SAM tab (it will save to your project)
3. **Go to DreamEdit3D Training tab**
4. **Click "📁 Load from Current Project"** button
5. **Select which view** to use (1, 2, 3, or 4)
6. Done! ✅

**Benefits:**
- ✅ One-click loading
- ✅ Automatically finds SAM masks from your project
- ✅ No need to remember folder names
- ✅ Works with the project system

**Example:**
```
1. Create project: "my_chicken_model"
2. SAM Tab → Process masks → Saves to: projects/my_chicken_model/01_sam_masks/
3. DreamEdit3D Tab → Click "Load from Current Project" → Loads automatically!
```

### Method 2: Load from Folder Dropdown

**For advanced users or loading from other sources**

The dropdown now shows:
- **Project folders**: Labeled as "Project: project_name"
- **Legacy folders**: Labeled as "Legacy: folder_name"

**Steps:**
1. Open the **"Option 2: Load from folder"** dropdown
2. You'll see entries like:
   - `Project: my_chicken_2025_01_28_143022` ← From project system
   - `Legacy: masked_output_1761576640` ← Old system folders
3. Select one
4. Click "Load Folder"

## 📋 Complete Workflow Example

```
Step 1: Create Project
→ Top of page → Enter "my_3d_model" → Click "Create New Project"

Step 2: SAM Segmentation
→ SAM Tab → Brush Masking → Draw masks → Process
✅ Saved to: projects/my_3d_model_2025_01_28_143022/01_sam_masks/

Step 3: Load in DreamEdit3D
→ DreamEdit3D Tab → Click "📁 Load from Current Project"
✅ Automatically loads masks from your project!

Step 4: Train
→ Fill in concept names → Click "Start Training"
✅ Model saved to: projects/my_3d_model_2025_01_28_143022/02_trained_model/

Step 5: Generate
→ Select model → Enter prompt → Generate
✅ Image saved to: projects/my_3d_model_2025_01_28_143022/03_multiview_images/
```

## 🔍 Troubleshooting

**❌ "No project selected"**
- Make sure you created or selected a project at the top of the page

**❌ "No SAM masks found in project"**
- Run SAM segmentation first (Brush Masking tab)
- Check that files are in `{project}/01_sam_masks/view_*/`

**❌ Button not loading anything**
- Check the project has `01_sam_masks` folder with view subdirectories
- Refresh the page if needed

## 💡 Pro Tips

1. **Always create a project first** before starting any work
2. **Use "Load from Current Project"** for easiest workflow
3. The **dropdown shows all available sources** (projects + legacy)
4. You can **switch between projects** using the dropdown at the top
5. **View selector** lets you pick which camera view to train on

## 🎓 Why This is Better

**Old Way:**
```
1. SAM → Save to random folder
2. Remember folder name
3. DreamEdit3D → Type in path manually
4. Hope you got it right
```

**New Way:**
```
1. Create project once
2. SAM → Auto-saves to project
3. DreamEdit3D → Click one button
4. Everything organized automatically! ✅
```

Enjoy the streamlined workflow! 🎨
