# GPT-4V Automatic Concept Naming

## Overview

The DreamEdit3D training pipeline now supports **automatic object detection** using GPT-4V! When loading masks for training, the system will automatically detect what objects are in each mask and fill in the "Concept Names" field.

## How It Works

### Before (Manual):
1. Create masks using SAM
2. Load masks for training
3. **Manually type** concept names: "tomato, bowl, creature"
4. Train model

### After (Automatic):
1. Create masks using SAM
2. Set OpenAI API key: `export OPENAI_API_KEY=your_key_here`
3. Load masks for training
4. **GPT-4V automatically detects** and fills: "tomato, bowl, creature"
5. Review/edit if needed
6. Train model

## Setup

### Install OpenAI Package
```bash
pip install openai
```

### Set API Key
```bash
# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here

# Or add to ~/.bashrc for permanent setup
echo 'export OPENAI_API_KEY=sk-your-api-key-here' >> ~/.bashrc
source ~/.bashrc
```

### Windows
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

## Usage

### In Gradio App

1. **Create masks** using SAM as usual
2. **Set your OpenAI API key** (see above)
3. Go to "DreamEdit3D Training" tab
4. Click **"Load from Project"**
5. Watch the console output:
   ```
   🤖 Attempting GPT-4V auto-detection for concept names...
     Detecting object 1/2...
     ✓ GPT-4V detected: 'tomato'
     Detecting object 2/2...
     ✓ GPT-4V detected: 'bowl'
   ✓ Auto-detected: tomato, bowl
   ```
6. The **"Concept Names"** field will be automatically filled!
7. Review the names (you can edit them if needed)
8. Click "Train Model"

### Fallback Behavior

The system is robust and will fallback gracefully:

1. **✓ Best case**: GPT-4V detects all objects → Auto-fills concept names
2. **⚠ No API key**: Uses generic "object, object1" → Can edit manually
3. **⚠ API error**: Falls back to generic names → Can edit manually
4. **⚠ OpenAI not installed**: Falls back to generic names → Can edit manually

**You always have the option to manually edit the concept names before training!**

## Cost

GPT-4V API calls:
- **Model**: `gpt-4o` (cheaper than gpt-4-vision-preview)
- **Cost**: ~$0.001-0.002 per image/mask
- **Example**: 2 masks = ~$0.002-0.004 per project

Very affordable for occasional use!

## Examples

### Example 1: Tomato + Bowl
```
Input masks: mask0.png, mask1.png
Auto-detected: "tomato, bowl"
```

### Example 2: Character Parts
```
Input masks: mask0.png, mask1.png, mask2.png
Auto-detected: "body, head, accessory"
```

### Example 3: Vehicle
```
Input masks: mask0.png
Auto-detected: "car"
```

## Troubleshooting

### No auto-detection happening

Check console output for warnings:

```bash
# Should see:
🤖 Attempting GPT-4V auto-detection for concept names...

# If you see:
⚠ No OpenAI API key found, using fallback: object
→ Set your OPENAI_API_KEY environment variable

# If you see:
⚠ OpenAI not available, using fallback: object
→ Install openai: pip install openai
```

### Detection is wrong

You can always **manually edit** the concept names in the text box before training!

### Want to disable auto-detection

Simply don't set the `OPENAI_API_KEY` environment variable. The system will fall back to generic names.

## Technical Details

- **Detection code**: `/utils/gpt_object_detector.py`
- **Integration**: `/main.py` (lines ~3558-3573)
- **How it works**:
  1. Applies mask to image (isolates object)
  2. Sends masked image to GPT-4o
  3. Asks: "What is this object?"
  4. Returns simple name (e.g., "tomato", "car")
  5. Auto-fills concept names field

## Benefits

- ⏱️ **Saves time**: No more typing object names
- 🎯 **More accurate**: Consistent naming across projects
- 🧠 **Smart**: GPT-4V understands context
- 🔄 **Safe**: Always falls back to manual if needed

---

**Enjoy automatic concept detection!** 🎉
