RULES = [

    # ==========================================================
    # ĐÁP ÁN
    # ==========================================================

    {
        "name": "answer_question",
        "description": "Hỏi đáp án câu hỏi",
        "contains_all": ["đáp án", "câu"],
        "score": 45,
        "severity": "high"
    },

    {
        "name": "read_answer",
        "description": "Yêu cầu đọc đáp án",
        "contains_all": ["đọc", "đáp án"],
        "score": 50,
        "severity": "high"
    },

    {
        "name": "say_answer",
        "description": "Yêu cầu nói đáp án",
        "contains_all": ["nói", "đáp án"],
        "score": 50,
        "severity": "high"
    },

    {
        "name": "find_answer",
        "description": "Tìm đáp án",
        "contains_all": ["tìm", "đáp án"],
        "score": 45,
        "severity": "high"
    },

    {
        "name": "correct_answer",
        "description": "Xác nhận đáp án",
        "contains_any": [
            "đúng không",
            "đúng chưa",
            "đúng hông",
            "đúng hả"
        ],
        "score": 20,
        "severity": "medium"
    },

    {
        "name": "choice_answer",
        "description": "Hỏi lựa chọn A B C D",
        "contains_any": [
            "a hay b",
            "b hay c",
            "c hay d",
            "a hay c",
            "a hay d",
            "b hay d"
        ],
        "score": 25,
        "severity": "medium"
    },

    # ==========================================================
    # XEM BÀI
    # ==========================================================

    {
        "name": "show_exam",
        "description": "Xin xem bài",
        "contains_all": ["cho", "xem", "bài"],
        "score": 60,
        "severity": "high"
    },

    {
        "name": "look_exam",
        "description": "Xin coi bài",
        "contains_all": ["coi", "bài"],
        "score": 50,
        "severity": "high"
    },

    {
        "name": "send_exam",
        "description": "Gửi bài",
        "contains_all": ["gửi", "bài"],
        "score": 60,
        "severity": "high"
    },

    {
        "name": "show_answer",
        "description": "Cho xem đáp án",
        "contains_all": ["xem", "đáp án"],
        "score": 60,
        "severity": "high"
    },

    {
        "name": "send_answer",
        "description": "Gửi đáp án",
        "contains_all": ["gửi", "đáp án"],
        "score": 60,
        "severity": "high"
    },

    {
        "name": "tilt_exam",
        "description": "Nghiêng bài",
        "contains_all": ["nghiêng", "bài"],
        "score": 55,
        "severity": "high"
    },

    # ==========================================================
    # NHỜ GIÚP
    # ==========================================================

    {
        "name": "solve_help",
        "description": "Giải giúp",
        "contains_all": ["giải", "giúp"],
        "score": 45,
        "severity": "high"
    },

    {
        "name": "do_help",
        "description": "Làm giúp",
        "contains_all": ["làm", "giúp"],
        "score": 45,
        "severity": "high"
    },

    {
        "name": "do_for_me",
        "description": "Làm hộ",
        "contains_any": [
            "làm hộ",
            "làm dùm",
            "làm giùm"
        ],
        "score": 45,
        "severity": "high"
    },

    {
        "name": "hint",
        "description": "Xin gợi ý",
        "contains_any": [
            "gợi ý",
            "gợi ý đi",
            "hint"
        ],
        "score": 30,
        "severity": "medium"
    },

    {
        "name": "tell_me",
        "description": "Chỉ bài",
        "contains_any": [
            "chỉ tao",
            "chỉ mình",
            "chỉ tui"
        ],
        "score": 35,
        "severity": "medium"
    },

    # ==========================================================
    # AI
    # ==========================================================

    {
        "name": "ai_tools",
        "description": "Sử dụng AI",
        "contains_any": [
            "chatgpt",
            "chat gpt",
            "gemini",
            "copilot",
            "claude",
            "deepseek",
            "perplexity"
        ],
        "score": 70,
        "severity": "high"
    },

    # ==========================================================
    # GOOGLE
    # ==========================================================

    {
        "name": "google",
        "description": "Tra Google",
        "contains_any": [
            "google",
            "tra google",
            "google giúp",
            "mở google",
            "tìm google"
        ],
        "score": 60,
        "severity": "high"
    },

    # ==========================================================
    # ĐỜI THƯỜNG
    # ==========================================================

    {
        "name": "casual_cheating",
        "description": "Ngôn ngữ đời thường",
        "contains_any": [
            "ê",
            "ê mày",
            "ê tao hỏi",
            "ê chỉ tao",
            "ê đáp án",
            "mày biết không",
            "mày làm tới đâu",
            "mày chọn gì",
            "mày chọn câu mấy",
            "mày làm được không",
            "câu này sao",
            "câu này gì",
            "câu mấy",
            "câu bao nhiêu",
            "là a hả",
            "là b hả",
            "là c hả",
            "là d hả",
            "đáp án gì",
            "đáp án sao",
            "cho tao coi",
            "cho tao xem",
            "cho coi tí",
            "cho xem tí",
            "cho nhìn tí",
            "coi xíu",
            "xem xíu",
            "xem phát",
            "đọc đi",
            "nói đi",
            "nói nhỏ",
            "nói khẽ",
            "đọc nhỏ",
            "đọc khẽ"
        ],
        "score": 15,
        "severity": "low"
    }

]