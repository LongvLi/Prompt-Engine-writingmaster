import argparse
import json
import os
import sys
import logging
from prompt_engine import PromptGenerationEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_argparse():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description='提示词生成引擎命令行工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 生成提示词命令
    generate_parser = subparsers.add_parser('generate', help='生成提示词')
    generate_parser.add_argument('-i', '--input', type=str, help='输入文本文件路径')
    generate_parser.add_argument('-o', '--output', type=str, help='输出JSON文件路径')
    generate_parser.add_argument('--text', type=str, help='直接输入文本内容')

    # 添加案例命令
    add_parser = subparsers.add_parser('add', help='添加案例到库')
    add_parser.add_argument('-i', '--input', type=str, required=True, help='JSON格式案例文件路径')

    # 反馈命令
    feedback_parser = subparsers.add_parser('feedback', help='提供反馈')
    feedback_parser.add_argument('-i', '--input', type=str, required=True, help='原始生成结果JSON文件路径')
    feedback_parser.add_argument('-a', '--adjusted', type=str, required=True, help='调整后的提示词JSON文件路径')
    feedback_parser.add_argument('-r', '--rating', type=int, required=True, choices=range(1, 6), help='评分(1-5)')

    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理文本文件')
    batch_parser.add_argument('-d', '--directory', type=str, required=True, help='输入文本文件目录')
    batch_parser.add_argument('-o', '--output', type=str, required=True, help='输出JSON文件目录')

    return parser


def read_text_file(file_path):
    """读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        return None


def write_json_file(data, file_path):
    """写入JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"已写入文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"写入文件失败: {str(e)}")
        return False


def handle_generate(args, engine):
    """处理生成提示词命令"""
    # 获取输入文本
    text = None
    if args.input:
        text = read_text_file(args.input)
    elif args.text:
        text = args.text
    else:
        print("请输入文本内容(输入EOF结束):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            text = '\n'.join(lines)

    if not text:
        logger.error("未提供有效的输入文本")
        return

    # 生成提示词
    result = engine.generate_prompts(text)

    # 输出结果
    if args.output:
        write_json_file(result, args.output)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_add_case(args, engine):
    """处理添加案例命令"""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            case = json.load(f)

        if "正文内容" not in case or "段落句式优秀提示词" not in case or "生成要求优秀提示词" not in case:
            logger.error("JSON文件格式不符合要求")
            return

        engine.add_case(
            case["正文内容"],
            case["段落句式优秀提示词"],
            case["生成要求优秀提示词"]
        )
        logger.info("已成功添加案例")

    except Exception as e:
        logger.error(f"添加案例失败: {str(e)}")


def handle_feedback(args, engine):
    """处理反馈命令"""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            original = json.load(f)

        with open(args.adjusted, 'r', encoding='utf-8') as f:
            adjusted = json.load(f)

        # 处理反馈
        engine.process_feedback(
            content=original["正文内容"],
            original_prompts={
                "段落句式": original["段落句式"],
                "生成要求": original["生成要求"]
            },
            adjusted_prompts={
                "段落句式": adjusted["段落句式"],
                "生成要求": adjusted["生成要求"]
            },
            success_rating=args.rating
        )

        logger.info(f"已处理反馈，评分: {args.rating}")

    except Exception as e:
        logger.error(f"处理反馈失败: {str(e)}")


def handle_batch(args, engine):
    """处理批量生成命令"""
    try:
        if not os.path.exists(args.directory):
            logger.error(f"目录不存在: {args.directory}")
            return

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        file_count = 0
        success_count = 0

        for filename in os.listdir(args.directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(args.directory, filename)
                text = read_text_file(file_path)

                if text:
                    file_count += 1
                    result = engine.generate_prompts(text)

                    output_file = os.path.join(args.output, filename.replace('.txt', '.json'))
                    if write_json_file(result, output_file):
                        success_count += 1

        logger.info(f"批量处理完成: 共 {file_count} 个文件, 成功 {success_count} 个")

    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")


def main():
    """主函数"""
    parser = setup_argparse()
    args = parser.parse_args()

    # 初始化引擎
    engine = PromptGenerationEngine()

    if args.command == 'generate':
        handle_generate(args, engine)
    elif args.command == 'add':
        handle_add_case(args, engine)
    elif args.command == 'feedback':
        handle_feedback(args, engine)
    elif args.command == 'batch':
        handle_batch(args, engine)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()