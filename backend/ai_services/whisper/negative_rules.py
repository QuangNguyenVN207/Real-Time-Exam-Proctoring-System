NEGATIVE_RULES = [

    # ==================================================
    # Giáo viên - sinh viên
    # ==================================================

    {
        "name": "ask_teacher",

        "contains_any": [

            "thầy cho em hỏi",
            "cô cho em hỏi",
            "em hỏi thầy",
            "em hỏi cô",
            "em chưa hiểu"

        ],

        "penalty": 60
    },

    {
        "name": "permission",

        "contains_any": [

            "em xin phép",
            "xin phép",
            "cho em xin phép",
            "em ra ngoài",
            "em vào lại"

        ],

        "penalty": 60
    },

    {
        "name": "submit",

        "contains_any": [

            "em nộp bài",
            "nộp bài",
            "em làm xong",
            "làm xong rồi",
            "em hoàn thành",
            "hoàn thành rồi"

        ],

        "penalty": 60
    },

    {
        "name": "greeting",

        "contains_any": [

            "chào thầy",
            "chào cô",
            "cảm ơn thầy",
            "cảm ơn cô",
            "em cảm ơn",
            "xin chào"

        ],

        "penalty": 70
    },

    # ==================================================
    # Trao đổi bình thường
    # ==================================================

    {
        "name": "normal_exam",

        "contains_any": [

            "bắt đầu làm bài",
            "đã rõ",
            "vâng",
            "dạ",
            "ok",
            "được rồi"

        ],

        "penalty": 40
    },

    # ==================================================
    # Tự nói với bản thân
    # ==================================================

    {
        "name": "thinking",

        "contains_any": [

            "để xem",
            "để mình nghĩ",
            "mình nghĩ",
            "hình như",
            "có lẽ",
            "chắc là"

        ],

        "penalty": 20
    },

    # ==================================================
    # Không phải gian lận
    # ==================================================

    {
        "name": "technical",

        "contains_any": [

            "micro",
            "mic",
            "loa",
            "âm thanh",
            "mạng lag",
            "không nghe"

        ],

        "penalty": 30
    }

]