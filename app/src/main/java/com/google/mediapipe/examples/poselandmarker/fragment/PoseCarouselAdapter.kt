package com.google.mediapipe.examples.poselandmarker.fragment

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.cardview.widget.CardView
import androidx.recyclerview.widget.RecyclerView
import com.google.mediapipe.examples.poselandmarker.R

class PoseCarouselAdapter(
    private val poseNames: List<String>,
    private val onPoseSelected: (String) -> Unit
) : RecyclerView.Adapter<PoseCarouselAdapter.PoseViewHolder>() {

    private var selectedPosition = 0

    inner class PoseViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val cardView: CardView = view as CardView
        val nameText: TextView = view.findViewById(R.id.pose_name_text)

        init {
            view.setOnClickListener {
                val position = bindingAdapterPosition
                if (position != RecyclerView.NO_POSITION && position != selectedPosition) {
                    val previous = selectedPosition
                    selectedPosition = position
                    notifyItemChanged(previous)
                    notifyItemChanged(selectedPosition)
                    onPoseSelected(poseNames[position])
                }
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PoseViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_pose_thumbnail, parent, false)
        return PoseViewHolder(view)
    }

    override fun onBindViewHolder(holder: PoseViewHolder, position: Int) {
        holder.nameText.text = poseNames[position]
        
        if (position == selectedPosition) {
            holder.cardView.setCardBackgroundColor(Color.parseColor("#4CAF50")) // Green for selected
            holder.nameText.setTextColor(Color.WHITE)
        } else {
            holder.cardView.setCardBackgroundColor(Color.parseColor("#80000000")) // Semi-transparent black
            holder.nameText.setTextColor(Color.LTGRAY)
        }
    }

    override fun getItemCount() = poseNames.size
}
