from PySide6 import QtCore, QtGui
import imkit as imk
import numpy as np
from .text.text_item_properties import TextItemProperties
from .text_item import OutlineInfo, OutlineType  # Import definitions only if needed, mostly for logic

class ImageSaveRenderer:
    def __init__(self, image: np.ndarray):
        self.rgb_image = image
        self.qimage = self.img_array_to_qimage(image)
        self.text_items_state = []
        self.patches = []

    def img_array_to_qimage(self, rgb_img: np.ndarray) -> QtGui.QImage:
        height, width, channel = rgb_img.shape
        bytes_per_line = channel * width
        # Copy data to avoid memory issues when the numpy array is garbage collected
        return QtGui.QImage(rgb_img.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()

    def add_state_to_image(self, state, page_idx=None, main_page=None):
        # Create a deep copy of the state to avoid modifying the original during spanning processing
        # effectively doing what add_spanning_text_items does but purely on data
        local_state = state.copy()
        
        # Add spanning text items if we have the context to do so
        if page_idx is not None and main_page is not None:
            self.add_spanning_text_items(local_state, page_idx, main_page)

        self.text_items_state.extend(local_state.get('text_items_state', []))

    def add_spanning_text_items(self, viewer_state, page_idx, main_page):
        """
        Add text items from spanning blocks that should appear on this page.
        """
        existing_text_items = viewer_state.get('text_items_state', [])
        # Ensure we are working with a list we can append to
        if existing_text_items is None:
            existing_text_items = []

        current_image_path = main_page.image_files[page_idx]
        current_image = imk.read_image(current_image_path)
        current_page_height = current_image.shape[0]

        for other_page_idx, other_image_path in enumerate(main_page.image_files):
            if other_page_idx == page_idx:
                continue

            if other_image_path not in main_page.image_states:
                continue

            page_gap = page_idx - other_page_idx
            if abs(page_gap) != 1:  # Only check adjacent pages
                continue

            # other_image = imk.read_image(other_image_path) # Not strictly needed if we trust height in state? 
            # Actually we need height. Assuming we can read it fast or it's cached.
            # Ideally this info should be passed in, but logic is preserved from original.
            other_image = imk.read_image(other_image_path)
            other_page_height = other_image.shape[0]
            
            other_viewer_state = main_page.image_states[other_image_path].get('viewer_state', {})
            other_text_items = other_viewer_state.get('text_items_state', [])

            if not other_text_items:
                continue

            for text_item in other_text_items:
                pos = text_item.get('position', (0, 0))
                item_x1, item_y1 = pos
                height = text_item.get('height', 0)
                item_y2 = item_y1 + height

                new_pos = None

                if page_gap == 1:  # Current page is BELOW other page
                    if item_y2 > other_page_height:
                        new_y = -(other_page_height - item_y1)
                        new_pos = (item_x1, new_y)

                elif page_gap == -1:  # Current page is ABOVE other page
                    if item_y1 < 0:
                        new_y = current_page_height + item_y1
                        new_pos = (item_x1, new_y)

                if new_pos:
                    spanning_text_item = text_item.copy()
                    spanning_text_item['position'] = new_pos
                    existing_text_items.append(spanning_text_item)

        viewer_state['text_items_state'] = existing_text_items

    def apply_patches(self, patches: list[dict]):
        self.patches = patches

    def render_to_image(self):
        # 1. Setup Canvas (Native Resolution for crisp text)
        # Using scale_factor > 1 causes blur due to downscaling interpolation. 
        # Native 1:1 rendering produces sharper edges for manga text.
        scale_factor = 1.0 
        original_size = self.qimage.size()
        scaled_size = original_size * scale_factor # effectively original size
        
        final_image = QtGui.QImage(scaled_size, QtGui.QImage.Format.Format_ARGB32)
        final_image.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(final_image)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

            # 2. Draw Original Image Background
            # Scale background to match high-res canvas (1:1 now)
            target_rect = QtCore.QRectF(0, 0, scaled_size.width(), scaled_size.height())
            painter.drawImage(target_rect, self.qimage, QtCore.QRectF(self.qimage.rect()))

            # 3. Draw Patches
            # Patches coordinate are in original image space, need to be scaled
            for patch in self.patches:
                x, y, w, h = patch['bbox']
                if 'png_path' in patch:
                    patch_img_arr = imk.read_image(patch['png_path'])
                else:
                    patch_img_arr = patch['image']

                patch_qimg = self.img_array_to_qimage(patch_img_arr)
                
                # Destination rect in scaled canvas
                dest_rect = QtCore.QRectF(
                    x * scale_factor, 
                    y * scale_factor, 
                    w * scale_factor, 
                    h * scale_factor
                )
                painter.drawImage(dest_rect, patch_qimg)

            # 4. Draw Text Items
            for text_block_dict in self.text_items_state:
                self._draw_text_item(painter, text_block_dict, scale_factor)

        finally:
            painter.end()

        # 5. Downscale back to original size - SKIPPED to preserve sharpness
        # final_image = final_image.scaled(original_size, ...) 
        if scale_factor != 1.0:
             final_image = final_image.scaled(original_size, 
                                 QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                                 QtCore.Qt.TransformationMode.SmoothTransformation)

        # 6. Return QImage directly to avoid numpy conversion loss and overhead
        # final_image = final_image.convertToFormat(QtGui.QImage.Format.Format_RGB888) -- Not needed for direct save
        return final_image

    def _draw_text_item(self, painter: QtGui.QPainter, text_block_dict: dict, scale: float):
        props = TextItemProperties.from_dict(text_block_dict)
        
        # Prepare QTextDocument
        doc = QtGui.QTextDocument()
        
        # Setup Font
        font = QtGui.QFont(props.font_family, props.font_size * scale) # Scale font size!
        font.setBold(props.bold)
        font.setItalic(props.italic)
        font.setUnderline(props.underline)
        
        # Setup Alignment
        text_option = QtGui.QTextOption()
        text_option.setAlignment(props.alignment)
        if props.direction:
             text_option.setTextDirection(props.direction)
        doc.setDefaultTextOption(text_option)
        doc.setDefaultFont(font) # Should affect formatting
        
        # Handle HTML vs Plain Text
        is_html = self._is_html(props.text)
        
        # Set text and color (color handled via CSS/HTML or DefaultColor)
        # We need to ensure color is applied correctly.
        # QGraphicsTextItem.setDefaultTextColor works for plain text.
        # For HTML we assume color is in tags or we need to wrap it?
        # TextBlockItem puts color in char format.
        
        # NOTE: props.font_size is already in 'points' presumably. 
        # When scaling canvas by 2, we must scale font size by 2.
        
        # Apply CSS for line spacing if needed
        # QTextBlockFormat line spacing.
        
        # Build content with correct formatting
        cursor = QtGui.QTextCursor(doc)
        block_fmt = QtGui.QTextBlockFormat()
        block_fmt.setAlignment(props.alignment)
        
        if props.line_spacing:
            # ProportionalHeight: 100 = 1x.
            block_fmt.setLineHeight(float(props.line_spacing * 100), QtGui.QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        
        cursor.select(QtGui.QTextCursor.SelectionType.Document)
        cursor.mergeBlockFormat(block_fmt)
        
        # Set Text Content
        if is_html:
            doc.setHtml(props.text)
            # HTML usually overrides colors, but if not we might need to apply default color
            # But let's assume HTML has what it needs or we use default color
            # If doc.toPlainText() is used, we lose HTML.
            # We need to apply font/color to the document if not in HTML.
            # But setTextWidth needs to be set first to wrap correctly?
            pass
        else:
            doc.setPlainText(props.text)
            
            # Apply Color and Font to whole document
            cursor.select(QtGui.QTextCursor.SelectionType.Document)
            char_fmt = QtGui.QTextCharFormat()
            char_fmt.setForeground(props.text_color)
            char_fmt.setFont(font)
            cursor.mergeCharFormat(char_fmt)
            
        # Set Width to trigger Wrapping
        # props.width is in original coords. Scale it.
        doc.setTextWidth(props.width * scale)

        # Transformation Setup
        painter.save()
        
        # Move to position (scaled)
        pos_x, pos_y = props.position
        painter.translate(pos_x * scale, pos_y * scale)
        
        # Rotation
        # transform_origin is relative to item top-left in TextBlockItem usually?
        # TextBlockItem.setTransformOriginPoint sets it relative to item local coords.
        # TextBlockItem default origin is center.
        if props.transform_origin:
            origin_x, origin_y = props.transform_origin
            # Move origin to (0,0), rotate, move back
            painter.translate(origin_x * scale, origin_y * scale)
            painter.rotate(props.rotation)
            painter.translate(-origin_x * scale, -origin_y * scale)
        else:
            painter.rotate(props.rotation) # Default rotation around 0,0 locally if no origin
            
        # Scale (item scale, distinct from global canvas scale)
        painter.scale(props.scale, props.scale)

        # Draw Outline (if any)
        # selection_outlines usually has [OutlineInfo(Full_Document)] for global outline
        outlines = props.selection_outlines
        if outlines:
            # We need to draw the outline "behind" the text.
            # QGraphicsTextItem logic: 
            # 1. Draw outlines (offset copies of text with outline color)
            # 2. Draw normal text
            
            # We only support Full_Document outline elegantly here
            # TextBlockItem logic draws separate passes.
             
            for outline in outlines:
                # Safe Enum Comparison (convert to string to avoid identity issues across modules)
                type_str = str(outline.type)
                target_full = str(OutlineType.Full_Document)
                target_sel = str(OutlineType.Selection)

                # Loose check: if "full_document" is in the string representation
                is_full = "full_document" in type_str.lower() or "full_document" in target_full.lower() and outline.type == OutlineType.Full_Document
                is_sel = "selection" in type_str.lower()
                
                # Direct value check (most robust if Enum values are strings)
                if hasattr(outline.type, 'value'):
                    is_full = outline.type.value == OutlineType.Full_Document.value
                    is_sel = outline.type.value == OutlineType.Selection.value
                
                if is_full or is_sel:
                    # Create a clone for outline
                    outline_doc = doc.clone()
                    outline_cursor = QtGui.QTextCursor(outline_doc)
                    outline_cursor.select(QtGui.QTextCursor.SelectionType.Document)
                    
                    # 1. Clear color (Make transparent base)
                    clear_fmt = QtGui.QTextCharFormat()
                    clear_fmt.setForeground(QtGui.QColor(0, 0, 0, 0))
                    outline_cursor.mergeCharFormat(clear_fmt)
                    
                    # 2. Apply Outline Color
                    outline_fmt = QtGui.QTextCharFormat()
                    c = outline.color
                    if not c or not c.isValid():
                        c = QtGui.QColor(255, 255, 255)
                    
                    # Ensure full opacity for outline unless explicitly transparent
                    if c.alpha() == 0:
                        c.setAlpha(255)

                    outline_fmt.setForeground(c)
                    outline_cursor.mergeCharFormat(outline_fmt)
                    
                    # Draw offsets using filled-disc approach
                    # Covers all integer pixel positions within the radius
                    # for a continuous stroke at any outline width
                    o_width = outline.width * scale
                    r = max(1, int(o_width + 0.5))
                    r_sq = o_width * o_width
                    offsets = [
                        (dx, dy)
                        for dx in range(-r, r + 1)
                        for dy in range(-r, r + 1)
                        if (dx != 0 or dy != 0) and (dx * dx + dy * dy) <= r_sq
                    ]
                    
                    for dx, dy in offsets:
                        painter.save()
                        painter.translate(dx, dy)
                        outline_doc.drawContents(painter)
                        painter.restore()

        # Draw Main Text
        doc.drawContents(painter)
        
        painter.restore()

    def _is_html(self, text):
        import re
        return bool(re.search(r'<[^>]+>', text))

    def save_image(self, output_path: str, quality: int = 100):
        final_qimg = self.render_to_image()
        # Direct QImage save bypasses numpy/imkit complications and uses optimal Qt codecs
        final_qimg.save(output_path, quality=quality)



